#!/bin/bash
#SBATCH -A NAISS2024-22-838 -p alvis
#SBATCH --gpus-per-node=A40:1

export HF_HOME=$TMPDIR
export HF_DATASETS_CACHE=$TMPDIR
export CUDA_VISIBLE_DEVICES="0"

# Initialize parameters
model=$1
test_filename=$2
pred_filename=$3
language=$4
model_type=$5

python src/evaluation.py \
       --output_folder "evaluation/${model}" \
       --language $language \
       --test_set "${test_filename}" \
       --predictions "${pred_filename}" \
       --model_type $model_type
