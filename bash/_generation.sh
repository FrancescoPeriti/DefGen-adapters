#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH --gpus-per-node=A100:3

# Initialize global variables
export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_LAUNCH_BLOCKING=1

# Arguments
pretrained_model_name_or_path=$1
peft_model_name_or_path=$2
test_filename=$3
batch_size=$4

# Parameters
language="nl"
hugginface_token="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"
max_new_tokens=256 #128
repetition_penalty=1.2
length_penalty=1.2
max_time=7


if [ "$peft_model_name_or_path" == "None" ]; then
    output_dir="predictions/$(basename "$pretrained_model_name_or_path")"
else
    output_dir="predictions/$(echo "$peft_model_name_or_path" | cut -d'/' -f2)"
fi

echo "$output_dir"
echo "$pretrained_model_name_or_path"
echo "$peft_model_name_or_path"
echo "$test_filename"
echo "$batch_size"

python src/generation.py \
       --language "$language" \
       --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
       --peft_model_name_or_path "${peft_model_name_or_path}" \
       --hugginface_token "${hugginface_token}" \
       --test_filename "$test_filename" \
       --max_new_tokens ${max_new_tokens} \
       --output_dir "$output_dir" \
       --batch_size ${batch_size} \
       --max_time ${max_time} \
       --repetition_penalty ${repetition_penalty} \
       --length_penalty ${length_penalty}
