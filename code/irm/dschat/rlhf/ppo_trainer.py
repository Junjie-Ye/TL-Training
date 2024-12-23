# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import json
import traceback
from pydoc import doc
from tokenize import Triple
from urllib import response
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import signal
from dschat.utils.utils import print_rank_0
import os
import concurrent.futures
import numpy
import re
import sys
torch.set_printoptions(threshold=50000)


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def test_answer_for_reward(pred_str, ans_str, test_data_type="mnli", functions=[], params={}, demo_action=None):
    if test_data_type == "mnli" or test_data_type == "snli":
        try:
            pattern = "###"

            pred = pred_str.split(pattern)
            gold = ans_str.split(pattern)

            if (len(pred) > 1):
                pred_answer = pred[1].strip().lower().replace('.', '')
                gold_answer = gold[1].strip().lower().replace('.', '')
                if pred_answer == gold_answer:
                    return 1
                else:
                    return 0
            else:
                return 0
        except Exception as e:
            return 0

    elif test_data_type == "boardgame":
        try:
            if pred_str.split("####")[-1].strip().lower() == ans_str.split("####")[-1].strip().lower():
                return 1
            else:
                return 0
        except Exception as e:
            return 0

    elif test_data_type == "raceHigh":
        try:
            pattern = "###"

            pred = pred_str.split(pattern)
            gold = ans_str.split(pattern)

            if (len(pred) > 1):
                pred_answer = pred[-1].strip().lower().replace('.', '')
                gold_answer = gold[-1].strip().lower().replace('.', '')
                # print('pred:', pred_answer)
                # print('gold:', gold_answer)
                # print('---\n\n')
                if pred_answer == gold_answer:
                    return 1
                else:
                    return 0
            else:
                return 0
        except Exception as e:
            return 0

    elif test_data_type == "tool_code":
        try:
            pattern = r'(.*?)\(([\s\S]*?)\)'
            pred_action = re.search(pattern, pred_str).group(1).strip()
            pred_action_input = re.search(pattern, pred_str).group(2)
            gold_action = re.search(pattern, ans_str).group(1).strip()
            gold_action_input = re.search(pattern, ans_str).group(2)

            dict_pattern = r"(\w+)\s*=\s*'([^']*)'"
            pred_matches = re.findall(dict_pattern, pred_action_input)
            gold_matches = re.findall(dict_pattern, gold_action_input)
            pred_dict = {key: value for key, value in pred_matches}
            gold_dict = {key: value for key, value in gold_matches}

            if pred_action not in functions and pred_action != "":
                return -2
            if pred_action != gold_action:
                return -1.5
            if demo_action == '':
                if pred_action in params:
                    p_hal = not set(pred_dict.keys()).issubset(
                        set(params[pred_action]))
                    p_redundant = any((item not in gold_dict.keys()) and (
                        item in params[pred_action]) for item in pred_dict.keys())
                else:
                    p_hal = False
                    p_redundant = False
            else:
                if demo_action in params:
                    p_hal = not set(pred_dict.keys()).issubset(
                        set(params[demo_action]))
                    p_redundant = any((item not in gold_dict.keys()) and (
                        item in params[demo_action]) for item in pred_dict.keys())
                else:
                    p_hal = False
                    p_redundant = False
            p_missing = any(item not in pred_dict.keys()
                            for item in gold_dict.keys())
            if p_hal and p_redundant and p_missing:
                return -1.8
            if p_hal and (p_redundant or p_missing):
                return -1.3
            if p_redundant and p_missing:
                return -1.0
            if p_hal:
                return -0.8
            if p_redundant or p_missing:
                return -0.5
            if pred_dict != gold_dict:
                return -0.25
            return 1

        except Exception as e:
            return -2


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor

        # self.actor_model.train()
        # self.actor_model.eval()

        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref

        # self.reward_model = self.rlhf_engine.reward

        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = 0.4
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0

    def _generate_sequence(self, prompts, mask, step, temperature=0.8, do_sample=False):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        if self.actor_model.module.config.model_type == "llama":
            # kwargs = dict(do_sample=False)
            kwargs = dict()
        else:
            kwargs = dict()

        # print(f"prompts:{self.tokenizer.batch_decode(prompts)}")
        with torch.no_grad():
            if self.args.test_data_type == "tool_code":
                eos_token_id = [2]
            print(f"eos_token_id: {eos_token_id}")
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                early_stopping=False,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=do_sample,
                # stopping_criteria=stop_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                synced_gpus=self.z3_enabled,
                **kwargs
            )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        if self.args.print_answers and (step % self.args.print_answers_interval == 0):

            # For Debug
            model_name = self.args.actor_model_name_or_path.split("/")[-1]
            import os

            folder_path = f"{self.args.output_dir}/results"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            filename = f'{folder_path}/step3_{model_name}.txt'

            with open(filename, 'a', encoding='utf-8') as file:
                p = f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts)}\n"
                a = f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans)}\n"

                file.write(p)
                file.write(a)

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                out_seq.append(seq[i:i + 1])

                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq, valid_ans_len.to(dtype=torch.float32)

    def generate_experience(self, prompts, mask, step,
                            answer=None,
                            temperature=0.8,
                            do_sample=False,
                            response_start=None
                            ):

        self.eval()
        generate_start = time.time()

        seq, ans_len = self._generate_sequence(
            prompts, mask, step, temperature=temperature, do_sample=do_sample)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)

            type_dict = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16
            }
            prompt_sentences = self.tokenizer.batch_decode(
                prompts, skip_special_tokens=True)
            seq_sentences = self.tokenizer.batch_decode(
                seq, skip_special_tokens=True)

            model_answer_sentences = []
            model_answer_reward_scores = []
            for i in range(len(prompts)):
                cur_seq = seq_sentences[i].split(response_start)
                model_answer_sentence_from_response_start = cur_seq[-1]
                if response_start in seq_sentences[i] and len(model_answer_sentence_from_response_start) > 0:
                    model_answer_sentence = model_answer_sentence_from_response_start
                else:
                    # model_answer_sentence_completion = seq_sentences[i][len(prompt_sentences[i])-1:]
                    model_answer_sentence_completion = seq_sentences[i][len(
                        prompt_sentences[i]):]
                    model_answer_sentence = model_answer_sentence_completion

                model_answer_sentences.append(model_answer_sentence)
                # print(f"prompt:{prompt_sentences[i]}")
                if self.args.test_data_type == "tool_code":
                    pattern = r"def\s+([^\s(]+)\((.*?)\):"
                    functions = re.findall(
                        pattern, prompt_sentences[i], re.DOTALL)
                    last_assistant_pos = prompt_sentences[i].rfind(
                        'Assistant:')
                    match_actions = re.search(
                        r"Assistant:(.*)", prompt_sentences[i][last_assistant_pos:], re.DOTALL)
                    demo_action = match_actions.group(1).strip()
                    function_names = []
                    function_params = {}
                    for function in functions:
                        function, params = function
                        function_names.append(function)
                        param_list = params.split(",") if params else []
                        params = []
                        for _p in param_list:
                            params += [p.strip()
                                       for p in re.findall(r"(.*?):.*?", _p, re.DOTALL)]
                        function_params[function] = params
                
                # print(f"function_names:{function_names}")
                # print(f"demo_action:{demo_action}")
                # print(f"function_params:{function_params}")
                # print(f"model_answer_sentence:{model_answer_sentence}")
                # print(f"answer:{answer[i]}")
                # print(
                #     f"test_answer_for_reward:{test_answer_for_reward(model_answer_sentence,answer[i],self.args.test_data_type,function_names,function_params,demo_action)}")
                model_answer_reward_scores.append(
                    test_answer_for_reward(
                        model_answer_sentence, answer[i], self.args.test_data_type, function_names, function_params, demo_action)
                )

            reward_score = torch.tensor(
                model_answer_reward_scores, dtype=type_dict[self.args.dtype])
            reward_score = reward_score.to(
                get_accelerator().current_device_name())
            # print(reward_score)

            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
            'ans_len': ans_len
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask, reward_last_token=1):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        if reward_last_token:
            ends = start + action_mask[:, start:].sum(1)
            # print("reward_last_token")

        else:
            ends = start + action_mask[:, start:].sum(1) + 1

        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():

            reward_last_token = self.args.reward_last_token
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask, reward_last_token=reward_last_token)
            if reward_last_token:
                ends = start + action_mask[:, start:].sum(1)
            else:
                ends = start + action_mask[:, start:].sum(1) + 1

            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

            avg_advantage = advantages.sum()/action_mask[:, start:].sum()
            avg_return = returns.sum()/action_mask[:, start:].sum()

        # process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])

        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]

        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss, avg_advantage, avg_return

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        # value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
