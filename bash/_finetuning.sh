#!/bin/bash
#SBATCH -A NAISS2024-22-838 -p alvis
#SBATCH --gpus-per-node=A40:4

#5-148

export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Arguments
base_model_name=$1
tag=$2
language=$3

# Parameters
seed=42
dropout=0.1
output_dir="models"
lora_rank=256
lora_alpha=512
batch_size=40
max_seq_length=300
weight_decay=0.01
warmup_ratio=0.15
learning_rate=1e-4
num_train_epochs=50
early_stopping_patience=5
early_stopping_threshold=0.001
gradient_accumulation_steps=1
peft_model_name="LlamaDictionary"
hugginface_token="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"
#train_filename="train-dev-test/train_dbnary_${language}.jsonl"
#dev_filename="train-dev-test/dev_dbnary_${language}.jsonl"


# Check if the variable contains spaces
if [[ "$language" =~ \  ]]; then
    IFS=' ' read -r -a array <<< "$language"

    train_filename=()
    for lang in "${array[@]}"; do
	train_filename+=("train-dev-test/train_dbnary_${lang}.jsonl")
    done
    
    dev_filename=()
    for lang in "${array[@]}"; do
	dev_filename+=("train-dev-test/dev_dbnary_${lang}.jsonl")
    done
else
    train_filename="train-dev-test/train_dbnary_${language}.jsonl"
    dev_filename="train-dev-test/dev_dbnary_${language}.jsonl"
fi

echo "${train_filename[@]}"
echo "${language[@]}"

python src/finetuning.py \
       --tag $tag \
       --language "${language[@]}" \
       --base_model_name $base_model_name \
       --hugginface_token $hugginface_token \
       --train_filename "${train_filename[@]}" \
       --dev_filename "${dev_filename[@]}" \
       --lora \
       --lora_rank $lora_rank \
       --lora_alpha $lora_alpha \
       --lora_dropout $dropout \
       --seed $seed \
       --finetuned_model_name "${peft_model_name}-${language[@]}" \
       --output_dir $output_dir \
       --max_seq_length $max_seq_length \
       --verbose \
       --cache_dir $TMPDIR \
       --weight_decay $weight_decay \
       --warmup_ratio $warmup_ratio \
       --batch_size $batch_size \
       --learning_rate $learning_rate \
       --num_train_epochs $num_train_epochs \
       --gradient_accumulation_steps $gradient_accumulation_steps \
       --num_rows -1 \
       --early_stopping_patience $early_stopping_patience \
       --early_stopping_threshold $early_stopping_threshold

