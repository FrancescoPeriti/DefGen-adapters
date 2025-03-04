# Generation

slurm_folder="slurm-output/generation"
test_sets=("train-dev-test/test_seen_dbnary" "train-dev-test/test_unseen_dbnary")


## DUTCH - NL ##
#__________________________________________________________________#
language="nl"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    continue # Done

    echo "Filename: $filename"
    
    # Pre-trained
    sbatch --time 18:00:00 --job-name BVG7BU-generation --output=${slurm_folder}/$language/GEITje-7B-ultra/${filename}.out bash/_generation.sh BramVanroy/GEITje-7B-ultra "None" "$test_set" 20 $language
    sbatch --time 18:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 18:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 18:00:00 --job-name fBVG7BU-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_BVG7BU/${filename}.out bash/_generation.sh BramVanroy/GEITje-7B-ultra models/LlamaDictionary-${language}_BVG7BU/checkpoint-19614 "$test_set" 20 $language
    sbatch --time 18:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-25607 "$test_set" 20 $language
    sbatch --time 1-18:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-25110 "$test_set" 35 $language
done
#__________________________________________________________________#


## ITALIAN - IT ##
#__________________________________________________________________#
language="it"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    continue # Done
    
    echo "Filename: $filename"
    
    # Pre-trained
    sbatch --time 18:00:00 --job-name M7IV1-generation --output=${slurm_folder}/$language/Minerva-7B-instruct-v1.0/${filename}.out bash/_generation.sh sapienzanlp/Minerva-7B-instruct-v1.0 "None" "$test_set" 20 $language
    sbatch --time 1:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 1:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 18:00:00 --job-name fM7IV1-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_M7IV1/${filename}.out bash/_generation.sh sapienzanlp/Minerva-7B-instruct-v1.0 models/LlamaDictionary-${language}_M7IV1/checkpoint-1764 "$test_set" 20 $language
    sbatch --time 18:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-1890 "$test_set" 20 $language
    sbatch --time 1-18:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-1200 "$test_set" 35 $language
done
#__________________________________________________________________#


## SWEDISH - SV ##
#__________________________________________________________________#
language="sv"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 18:00:00 --job-name L38BI-generation --output=${slurm_folder}/$language/Llama-3-8B-instruct/${filename}.out bash/_generation.sh AI-Sweden-Models/Llama-3-8B-instruct "None" "$test_set" 20 $language
    sbatch --time 1:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 1:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 18:00:00 --job-name fL38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L38BI/${filename}.out bash/_generation.sh AI-Sweden-Models/Llama-3-8B-instruct models/LlamaDictionary-${language}_L38BI/checkpoint-9480 "$test_set" 20 $language
    sbatch --time 18:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-23667 "$test_set" 20 $language
    sbatch --time 1-18:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-21804 "$test_set" 35 $language
done
#__________________________________________________________________#
