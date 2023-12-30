# PEFT Comparison

Comparing a bunch of PEFT methods on T5 and LLaMA.

## Installation instructions

Notice that we require specifically `adapter-transformers` (right before the 2.0)

```bash
pip install -e .
cd adapters
pip install -e .
cd ..
python -m nltk.downloader punkt
```

If you are running LLaMA fine-tuning, you will probably also need

```bash
huggingface-cli login
```

## Usage example

```bash
export model_name="t5-large"
export dataset_name="super_glue"
export dataset_config_name="boolq"
export adapter_config_string="hf_loha"

export experiment_name="$model_name-$dataset_name-$dataset_config_name-$adapter_config_string"

python scripts/finetuning_seq2seq.py \
    --output_dir "results/$experiment_name"\
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --dataset_config_name $dataset_config_name \
    --adapter_config_string $adapter_config_string \
    --total_batch_size 32 \
    --per_device_train_batch_size 8 \
    --max_source_length 128 \
    --max_target_length 8 \
    --num_beams 5 \
    --learning_rate 2e-4 \
    --eval_every_steps 1000
```


## All valid method names for `adaper_config_string`:

```
"full_tuning"
"pfeiffer"
"houlsby"
"scaled_parallel"
"compacter"
"compacter++"
"prefix_tuning"
"prefix_tuning_flat"
"lora"
"hf_lora"      # huggingface PEFT implementation of lora
"hf_lora_all"  # apply LoRA to all layers
"hf_krona"
"ia3"
"mam"
"unipelt"
"ln_tuning"
"bitfit"  # not useful, because neither T5 nor LLaMA use biases
```

Not yet implemented, but expected:
```
"attn_tuning"
"relora"
```
