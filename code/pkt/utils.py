from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.nn as nn
from torch.nn import functional as F
import torch


class GCELoss(nn.Module):
    def __init__(self, ignore_index=-100, q=0.0):
        super(GCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.q = q

    def forward(self, tokenizer, logits, targets, weights, weights_idx, input_token, input_ids, index_ass, action, max_weight=9):
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        weights = weights[valid_idx].float()
        loss_ids = input_ids[valid_idx]
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            if logits.size(-1) == 1:
                pred = torch.sigmoid(logits)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(logits, dim=-1)
            pred = torch.gather(
                pred, dim=-1, index=torch.unsqueeze(targets, -1))
            loss = (1-pred**self.q) / self.q

        loss_token = []
        for item in loss_ids.view(-1):
            loss_token.append(tokenizer.convert_ids_to_tokens(item.item()))
        true_indices = []
        for index, value in enumerate(valid_idx[0]):
            if value.item() is True:
                true_indices.append(index)
        mask = torch.ones(loss.size(), dtype=torch.bool, device=loss.device)
        pre_indices = torch.tensor(
            weights_idx['pre'], dtype=torch.long, device=loss.device)
        para_indices = torch.tensor(
            weights_idx['para'], dtype=torch.long, device=loss.device)
        for i in range(pre_indices.size(0)):
            pre_indices[i] -= index_ass
        for i in range(para_indices.size(0)):
            para_indices[i] -= index_ass

        mask[pre_indices] = False
        mask[para_indices] = False
        loss_pre = loss[pre_indices]
        loss_para = loss[para_indices]
        loss_other = loss[mask]
        loss_pre_size = loss_pre.size()[0]
        loss_other_size = len(loss_other)
        loss_pre = loss_pre.view(-1).sum()
        loss_para = loss_para.view(-1).sum()
        loss_other = loss_other.view(-1).sum()

        if loss_pre_size < loss_other_size:
            loss_pre = loss_pre * min(loss_other_size / loss_pre_size, max_weight)
            weights[pre_indices] = min(loss_other_size / loss_pre_size, max_weight)
        loss = (loss_pre + loss_para + loss_other) / weights.sum()
        return loss


class ToolTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(model=model,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        def remove_identical_sublist(a, b):
            return [item for item in b if item != a]

        def longest_common_prefix_length(X, Y):
            min_length = min(len(X), len(Y))
            for i in range(min_length):
                if X[i] != Y[i]:
                    return i
                
            return min_length

        def find_longest_common_subsequence_length(a, b):
            tool_pre = []
            max_length = 0
            for item in b:
                current_length = longest_common_prefix_length(a, item)
                if current_length != 0:
                    tool_pre.append(current_length)

            return max_length, tool_pre


        def extract_parameters_with_indices(tokens, index):
            answer = tokens[index + 2:]
            indices = []
            split_indices = [i - 1 for i, token in enumerate(answer) if token in [
                "',", "')", ')', ')_', '()', '()_', "▁'')", "▁')", '")', '▁")', '▁"")', '.")', ')")']]
            indices = [item + len(tokens[:index + 2])
                        for item in split_indices]
            if indices == []:
                indices = [len(tokens) - 3]
                
            return indices

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            input_token = []
            # Flatten the tensor to 1D
            for item in inputs['input_ids'].view(-1):
                input_token.append(
                    self.tokenizer.convert_ids_to_tokens(item.item()))

            function_names = []
            start_index = [idx for idx, text in enumerate(
                input_token) if text == 'istant' and input_token[idx + 1] == ':'][-1]
            end_tokens = ['(', '()', '(_', '_(', '_()', '()_', '("']
            for token in end_tokens:
                try:
                    end_index = input_token.index(token, start_index)
                    break
                except:
                    pass
            golden = input_token[start_index + 2: end_index]
            for i in range(len(input_token)):
                if (input_token[i] == 'Function' or input_token[i] == '▁Function') and input_token[i + 1] == ':':
                    name_tokens = []
                    j = i + 4
                    while input_token[j] != '(' and input_token[j] != '()' and input_token[j] != '():':
                        name_tokens.append(input_token[j])
                        j += 1
                    function_names.append(name_tokens)

            function_names = remove_identical_sublist(golden, function_names)
            _, tool = find_longest_common_subsequence_length(
                golden, function_names)
            loss_function = GCELoss()
            weights_idx = {}
            pre = []
            index = [idx for idx, text in enumerate(
                input_token) if text == 'istant'][-1]

            params_idx = extract_parameters_with_indices(input_token, index)
            labels = torch.full_like(inputs['input_ids'], -100)
            labels[0, index + 1: -1] = inputs['input_ids'][0, index + 2:]
            weights = torch.full_like(labels, 1.0)
            pre.append(index + 1)
            if tool != []:
                for item in tool:
                    if index + 1 + item not in pre:
                        pre.append(index + 1 + item)
            weights_idx['pre'] = pre
            weights_idx['para'] = params_idx

            loss = loss_function(tokenizer=self.tokenizer, logits=outputs['logits'], targets=labels, weights=weights,
                                 weights_idx=weights_idx, input_token=input_token, input_ids=inputs['input_ids'], index_ass=index + 1, action=golden, max_weight=self.args.max_weight)

            return (loss, outputs) if return_outputs else loss
