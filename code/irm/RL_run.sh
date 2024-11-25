export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

task_name=""
test_data_type="tool_code"

data_split="0,0,10"
data_path=""
echo $data_path

temperature=0.7
do_sample=0
num_train_epochs=3
kl_ctl=0.3
reward_last_token=1
max_prompt_seq_len=8192
max_answer_seq_len=512
actor_zero_stage=2
critic_zero_stage=3
actor_learning_rate=2e-7
critic_learning_rate=1e-7

output_base="./models/${task_name}/"
log_base=./${task_name}/
mkdir $log_base

per_device_generation_batch_size=1
per_device_training_batch_size=1
gradient_accumulation_steps=1

# base model
actor_model_name_or_path=""
critic_model_name_or_path=""

output_path=${output_base}${task_name}
data_output_path=${output_path}/data_files
log_path=${log_base}${task_name}.log
echo $output_path
tensorboard_dir=${log_base}tensorboard_${task_name}/
echo $tensorboard_dir

mkdir -p $output_path

deepspeed \
    --master_port 39925 \
    --include localhost:0,1,2,3,4,5,6,7 \
    main.py \
    --test_data_type $test_data_type \
    --seed 1234 \
    --data_path $data_path \
    --data_split $data_split \
    --data_output_path $data_output_path \
    --actor_learning_rate $actor_learning_rate \
    --critic_learning_rate $critic_learning_rate \
    --max_prompt_seq_len $max_prompt_seq_len \
    --max_answer_seq_len $max_answer_seq_len \
    --print_answers \
    --per_device_generation_batch_size $per_device_generation_batch_size \
    --per_device_training_batch_size $per_device_training_batch_size \
    --do_sample $do_sample \
    --temperature $temperature \
    --kl_ctl $kl_ctl \
    --actor_model_name_or_path $actor_model_name_or_path \
    --reward_last_token $reward_last_token \
    --num_train_epochs $num_train_epochs \
    --critic_model_name_or_path  $critic_model_name_or_path \
    --actor_zero_stage $actor_zero_stage \
    --critic_zero_stage $critic_zero_stage \
    --num_padding_at_beginning 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --deepspeed  \
    --enable_hybrid_engine \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --actor_dropout 0.0 \
    --enable_tensorboard \
    --tensorboard_path $tensorboard_dir \
    --output_dir $output_path \
    >   $log_path 2>&1
