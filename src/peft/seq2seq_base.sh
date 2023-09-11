#!/bin/bash

echo "STARTING EXP..."

python finetuning_seq2seq.py \
    --seed 0 \
    --model_name_or_path "t5-base" \
    --dataset_name "cnn_dailymail" \
    --dataset_config_name "3.0.0" \
    --peft_method "lora" \
    --r 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 3 \
    --learning_rate 0.0001 \
    --lr_scheduler_type "linear" \
    --lr_scheduler_warmup_percent 0.06 \
    --weight_decay 0 \
    --num_train_epochs 1 \
    --eval_every_steps 10000 \
    --wandb_project "PEFT_comparison" \
    --source_prefix "summarize: " \

echo "FINISHED EXP"

echo "STARTING EXP..."

python finetuning_seq2seq.py \
    --seed 0 \
    --model_name_or_path "t5-base" \
    --dataset_name "cnn_dailymail" \
    --dataset_config_name "3.0.0" \
    --peft_method "ia_3" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 3 \
    --learning_rate 0.001 \
    --lr_scheduler_type "linear" \
    --lr_scheduler_warmup_percent 0.06 \
    --weight_decay 0 \
    --num_train_epochs 1 \
    --eval_every_steps 10000 \
    --wandb_project "PEFT_comparison" \
    --source_prefix "summarize: " \

echo "FINISHED EXP"


echo "STARTING EXP..."

python finetuning_seq2seq.py \
    --seed 0 \
    --model_name_or_path "t5-base" \
    --dataset_name "cnn_dailymail" \
    --dataset_config_name "3.0.0" \
    --peft_method "prompt_tuning" \
    --num_virtual_tokens 20 \
    --prompt_tuning_init "text" \
    --prompt_tuning_init_text "summarize the following document " \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 3 \
    --learning_rate 0.001 \
    --lr_scheduler_type "linear" \
    --lr_scheduler_warmup_percent 0.06 \
    --weight_decay 0 \
    --num_train_epochs 1 \
    --eval_every_steps 10000 \
    --wandb_project "PEFT_comparison" \
    --source_prefix "summarize: " \

echo "FINISHED EXP"



echo "STARTING EXP..."

python finetuning_seq2seq.py \
    --seed 0 \
    --model_name_or_path "t5-base" \
    --dataset_name "cnn_dailymail" \
    --dataset_config_name "3.0.0" \
    --peft_method "prefix_tuning" \
    --num_virtual_tokens 20 \
    --prefix_projection true \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 3 \
    --learning_rate 0.0001 \
    --lr_scheduler_type "linear" \
    --lr_scheduler_warmup_percent 0.06 \
    --weight_decay 0 \
    --num_train_epochs 1 \
    --eval_every_steps 10000 \
    --wandb_project "PEFT_comparison" \
    --source_prefix "summarize: " \

echo "FINISHED EXP"

