slurm_folder="slurm-output/finetuning"

# For all languages, we sample 13k examples in the src/finetuning.py script
# We fine-tune Meta-Llama-3-8B-Instruct

# Romance (R)
languages="it es fr" # we use 'la pt ca' for evaluation
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# Germanic (G)
languages="sv de en" # we use 'da no nl' for evaluation
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# Slavic (S)
languages="pl ru"
sbatch --time 1-15:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# (G+R)
languages="it es sv de fr en" # we use 'la pt ca da no nl' for evaluation
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# This is an additional configuration
languages="ja fi ru" 
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# (All)
languages="it es tr ja el fi ku pl sv ru de fr en" # 13k
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# (R+G+S)
languages="it es pl sv ru de fr en" # we use 'la pt ca da no nl' for evaluation
sbatch --time 2-13:00:00 --job-name RGS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# (R+S)
languages="it es pl ru fr" # we use 'la pt ca' for evaluation
sbatch --time 23:00:00 --job-name RS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"

# (G+S)
languages="pl sv ru de en" # we use 'da no nl' for evaluation
sbatch --time 23:00:00 --job-name GS-finetuning --output="${slurm_folder}/${languages}/Meta-Llama-3-8B-Instruct.out" bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "${languages}"
