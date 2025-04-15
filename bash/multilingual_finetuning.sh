# CROSS-LINGUAL TRANSFER among typologically similar languages
slurm_folder="slurm-output/finetuning"

# Romance
languages="it es fr" # la pt ca -> 13k
#sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"
#bash bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"


# Germanic
languages="sv de en" # da no nl -> 13k (93k)
#sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# Slavic
languages="pl ru"
#sbatch --time 1-15:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# FEW-SHOT vs ZERO-SHOT trade-off
# few-shot la pt ca da no nl
# zero-shot the others
languages="it es sv de fr en" # -> 13k
#sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"


languages="ja fi ru" # el ku en -> 13k
#sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

languages="it es tr ja el fi ku pl sv ru de fr en" # 13k
#sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"


languages="it es pl sv ru de fr en" # -> 13k
sbatch --time 2-13:00:00 --job-name RGS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

languages="it es pl ru fr" # -> 13k
#sbatch --time 23:00:00 --job-name RS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

languages="pl sv ru de en" # -> 13k
#sbatch --time 23:00:00 --job-name GS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"
