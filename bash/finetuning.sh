# Finetuning

slurm_folder="slurm-output/finetuning"

## DUTCH - NL ##
#__________________________________________________________________#
language="nl"
sbatch --time 2-10:00:00 --job-name CL27BI-finetuning --output=${slurm_folder}/$language/ChocoLlama-2-7B-instruct.out bash/_finetuning.sh ChocoLlama/ChocoLlama-2-7B-instruct "_CL27BI" "$language"
sbatch --time 2-10:00:00 --job-name L3CL8BI-finetuning --output=${slurm_folder}/$language/Llama-3-ChocoLlama-8B-instruct.out bash/_finetuning.sh ChocoLlama/Llama-3-ChocoLlama-8B-instruct "_L3CL8BI" "$language"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
sbatch --time 2-10:00:00 --job-name BVG7BU-finetuning --output=${slurm_folder}/$language/GEITje-7B-ultra.out bash/_finetuning.sh BramVanroy/GEITje-7B-ultra "_BVG7BU" "$language"
#__________________________________________________________________#

## ITALIAN - IT ##
#__________________________________________________________________#
language="it"
sbatch --time 1-00:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 1-00:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
sbatch --time 1-00:00:00 --job-name M7BIV1-finetuning --output=${slurm_folder}/$language/Minerva-7B-instruct-v1.0.out bash/_finetuning[new].sh sapienzanlp/Minerva-7B-instruct-v1.0 "_M7BIV1" "$language"
#__________________________________________________________________#

## SWEDISH - SV ##
#__________________________________________________________________#
language="sv"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
sbatch --time 2-10:00:00 --job-name L38BI-finetuning --output=${slurm_folder}/$language/Llama-3-8B-instruct.out bash/_finetuning.sh AI-Sweden-Models/Llama-3-8B-instruct "_L38BI" "$language"
#__________________________________________________________________#

## NORWEGIAN - NO ##
#__________________________________________________________________#
language="no"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#

## JAPANESE - JA ##
#__________________________________________________________________#
language="ja"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#

## SPANISH - ES ##
#__________________________________________________________________#
language="es"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## GERMAN - DE ##
#__________________________________________________________________#
language="de"
sbatch --time 6-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 6-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## RUSSIAN - RU ##
#__________________________________________________________________#
language="ru"
sbatch --time 6-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 6-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## PORTOGUESE - PT ##
#__________________________________________________________________#
language="pt"
sbatch --time 1-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 1-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## GREEK - EL ##
#__________________________________________________________________#
language="el"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## FRENCH - FR ##
#__________________________________________________________________#
language="fr"
sbatch --time 6-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 6-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## TURKISH - TR ##
#__________________________________________________________________#
language="tr"
sbatch --time 1-15:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 1-15:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## MG ##
#__________________________________________________________________#
language="mg"
sbatch --time 15:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 15:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## DANISH - DA ##
#__________________________________________________________________#
language="da"
sbatch --time 3:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 3:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## CATALAN - CA ##
#__________________________________________________________________#
language="ca"
sbatch --time 2-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


##  - LT ##
#__________________________________________________________________#
language="lt"
sbatch --time 2:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


##  - LA ##
#__________________________________________________________________#
language="la"
sbatch --time 2:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 2:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


##  - PL ##
#__________________________________________________________________#
language="pl"
sbatch --time 3-05:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 3-05:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


##  - KU ##
#__________________________________________________________________#
language="ku"
sbatch --time 3:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 3:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


##  - ZH ##
#__________________________________________________________________#
language="zh"
sbatch --time 15:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 15:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## FINNISH - FI ##
#__________________________________________________________________#
language="fi"
sbatch --time 1-10:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 1-10:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#


## ENGLISH - EN ##
#__________________________________________________________________#
language="en"
sbatch --time 6-23:00:00 --job-name L27BCH-finetuning --output=${slurm_folder}/$language/Llama-2-7B-chat-hf.out bash/_finetuning.sh meta-llama/Llama-2-7b-chat-hf "_L27BCH" "$language"
sbatch --time 6-23:00:00 --job-name ML38BI-finetuning --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct.out bash/_finetuning.sh meta-llama/Meta-Llama-3-8B-Instruct "_ML38BI" "$language"
#__________________________________________________________________#
