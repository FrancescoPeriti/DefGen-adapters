# Generation

slurm_folder="slurm-output/generation"
test_sets=("train-dev-test/test_seen_dbnary" "train-dev-test/test_unseen_dbnary")


## DUTCH - NL ##
#__________________________________________________________________#
language="nl"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

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

    echo "Filename: $filename"
    
    # Pre-trained
    sbatch --time 18:00:00 --job-name M7BIV1-generation --output=${slurm_folder}/$language/Minerva-7B-instruct-v1.0/${filename}.out bash/_generation.sh sapienzanlp/Minerva-7B-instruct-v1.0 "None" "$test_set" 20 $language
    sbatch --time 1:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 1:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 18:00:00 --job-name fM7BIV1-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_M7BIV1/${filename}.out bash/_generation.sh sapienzanlp/Minerva-7B-instruct-v1.0 models/LlamaDictionary-${language}_M7BIV1/checkpoint-1764 "$test_set" 20 $language
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
    sbatch --time 18:00:00 --job-name L38BI-generation --output=${slurm_folder}/$language/Llama-3-8B-instruct/${filename}.out bash/_generation.sh AI-Sweden-Models/Llama-3-8B-instruct "None" "$test_set" 30 $language
    sbatch --time 18:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 20  $language
    sbatch --time 18:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 30 $language

    # Fine-tuned
    sbatch --time 18:00:00 --job-name fL38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L38BI/${filename}.out bash/_generation.sh AI-Sweden-Models/Llama-3-8B-instruct models/LlamaDictionary-${language}_L38BI/checkpoint-11112 "$test_set" 15 $language
    sbatch --time 18:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-25175 "$test_set" 15 $language
    sbatch --time 1-18:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-12038 "$test_set" 15 $language
done
#__________________________________________________________________#


## NORWEGIAN - NO ##
#__________________________________________________________________#
language="no"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 3:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 3:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-931 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-900 "$test_set" 35 $language
done
#__________________________________________________________________#


## SPANISH - ES ##
#__________________________________________________________________#
language="es"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 3:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 5:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-2610 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-3565 "$test_set" 35 $language
done
#__________________________________________________________________#


## JAPANESE - JA ##
#__________________________________________________________________#
language="ja"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 3:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 3:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-4578 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-2376 "$test_set" 35 $language
done
#__________________________________________________________________#


## GERMAN - DE ##
#__________________________________________________________________#
language="de"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 15:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 15:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 15:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-110048 "$test_set" 20 $language
    sbatch --time 15:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-64640 "$test_set" 35 $language
done
#__________________________________________________________________#


## PORTOGUESE - PT ##
#__________________________________________________________________#
language="pt"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 15:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 15:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 15:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-1666 "$test_set" 20 $language
    sbatch --time 15:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-1020 "$test_set" 35 $language
done
#__________________________________________________________________#


## GREEK - EL ##
#__________________________________________________________________#
language="el"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 15:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 15:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 15:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-20850 "$test_set" 20 $language
    sbatch --time 15:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-3552 "$test_set" 35 $language
done
#__________________________________________________________________#



## FRENCH - FR ##
#__________________________________________________________________#
language="fr"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-1666 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-1020 "$test_set" 35 $language
done
#__________________________________________________________________#


## TURKISH - TR #
#__________________________________________________________________#
language="tr"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-3536 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-2176 "$test_set" 35 $language
done
#__________________________________________________________________#


## CA ##
#__________________________________________________________________#
language="ca"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-3178 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-2769 "$test_set" 35 $language
done
#__________________________________________________________________#


## DA ##
#__________________________________________________________________#
language="da"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-66 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-85 "$test_set" 35 $language
done
#__________________________________________________________________#


## KU ##
#__________________________________________________________________#
language="ku"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    #sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    #sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-2468 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-2188 "$test_set" 27 $language
done
#__________________________________________________________________#


## PL ##
#__________________________________________________________________#
language="pl"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-11666 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-12804 "$test_set" 35 $language
done
#__________________________________________________________________#


## ZH ##
#__________________________________________________________________#
language="zh"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-1815 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-1470 "$test_set" 35 $language
done
#__________________________________________________________________#


## LA ##
#__________________________________________________________________#
language="la"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-38 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-38 "$test_set" 35 $language
done
#__________________________________________________________________#


## LT ##
#__________________________________________________________________#
language="lt"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-51 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-48 "$test_set" 35 $language
done
#__________________________________________________________________#


## MG ##
#__________________________________________________________________#
language="mg"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-750 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-980 "$test_set" 35 $language
done
#__________________________________________________________________#


## FINNISH - FI ##
#__________________________________________________________________#
language="fi"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 6:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-7128 "$test_set" 20 $language
    sbatch --time 6:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-5565 "$test_set" 35 $language
done
#__________________________________________________________________#


## RUSSIAN - RU ##
#__________________________________________________________________#
language="ru"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 10:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 10:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 10:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-17577 "$test_set" 20 $language
    sbatch --time 10:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-19776 "$test_set" 35 $language
done
#__________________________________________________________________#


## ENGLISH - EN ##
#__________________________________________________________________#
language="en"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 3-10:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 3-10:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 3-10:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-140926 "$test_set" 20 $language
    sbatch --time 3-10:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-128238 "$test_set" 35 $language
done
#__________________________________________________________________#


## FRENCH - FR ## 
#__________________________________________________________________#
language="fr"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"
    
    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 3-10:00:00 --job-name L27BCH-generation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf "None" "$test_set" 30  $language
    sbatch --time 3-10:00:00 --job-name ML38BI-generation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "None" "$test_set" 20 $language

    # Fine-tuned
    sbatch --time 3-10:00:00 --job-name fL27BCH-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_generation.sh meta-llama/Llama-2-7b-chat-hf models/LlamaDictionary-${language}_L27BCH/checkpoint-126661 "$test_set" 20 $language
    sbatch --time 3-10:00:00 --job-name fML38BI-generation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct models/LlamaDictionary-${language}_ML38BI/checkpoint-121100 "$test_set" 35 $language
done
#__________________________________________________________________#


#### CROSS-LINGUAL GENERATION using monolingual models
languages=("en" "fr" "de" "ru" "sv" "nl" "pl" "ku" "fi" "el" "ja" "ca" "es" "it" "tr" "pt" "zh" "no" "mg" "da" "lt" "la")

filename="train-dev-test/test_unseen_dbnary"

# Loop through languages
for lang1 in "${languages[@]}"; do
    for lang2 in "${languages[@]}"; do
        if [[ "$lang1" != "$lang2" ]]; then
	    if [[ -e "predictions/LlamaDictionary-${lang2}_ML38BI/test_unseen_dbnary_${lang1}.txt" ]]; then
	        continue
	    fi
	
	    model_checkpoint=$(ls -d models/LlamaDictionary-${lang2}_ML38BI/checkpoint-*)
	    echo $lang1 $lang2 $model_checkpoint
	    sbatch --time 10:00:00 --job-name ${lang1}_${lang2}-generation --output=${slurm_folder}/$lang1/LlamaDictionary-${lang2}_ML38BI/${filename}_${lang1}.out bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct ${model_checkpoint} "${filename}_${lang1}.jsonl" 25 $lang1
        fi
    done
done
