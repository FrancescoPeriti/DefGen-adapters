#sbatch --time 4-10:00:00 --job-name CL27BI-finetuning --output=slurm-output/finetuning/ChocoLlama-2-7B-instruct.out bash/_finetuning.sh ChocoLlama/ChocoLlama-2-7B-instruct "_CL27BI"
#sbatch --time 4-10:00:00 --job-name L3CL8BI-finetuning --output=slurm-output/finetuning/Llama-3-ChocoLlama-8B-instruct.out bash/_finetuning.sh ChocoLlama/Llama-3-ChocoLlama-8B-instruct "_L3CL8BI"
#sbatch --time 4-10:00:00 --job-name L27BCH-finetuning --output=slurm-output/finetuning/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH"
#sbatch --time 4-10:00:00 --job-name ML38BI-finetuning --output=slurm-output/finetuning/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI"
sbatch --time 4-10:00:00 --job-name BVG7BU-finetuning --output=slurm-output/finetuning/GEITje-7B-ultra.out bash/_finetuning.sh "BramVanroy/GEITje-7B-ultra" "_BVG7BU"


