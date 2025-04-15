# Evaluation

slurm_folder="slurm-output/evaluation"
test_sets=("train-dev-test/test_seen_dbnary" "train-dev-test/test_unseen_dbnary")

## DUTCH - NL ##
#__________________________________________________________________#
language="nl"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name BVG7BU-evaluation --output=${slurm_folder}/$language/GEITje-7B-ultra/${filename}.out bash/_evaluation.sh "GEITje-7B-ultra" "$test_set" "predictions/GEITje-7B-ultra/${filename}.txt" $language "GroNLP/bert-base-dutch-cased"
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "GroNLP/bert-base-dutch-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "GroNLP/bert-base-dutch-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fBVG7BU-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_BVG7BU/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_BVG7BU "$test_set" "predictions/LlamaDictionary-${language}_BVG7BU/${filename}_${language}.txt" $language "GroNLP/bert-base-dutch-cased"
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "GroNLP/bert-base-dutch-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "GroNLP/bert-base-dutch-cased"
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
    sbatch --time 6:00:00 --job-name M7BIV1-evaluation --output=${slurm_folder}/$language/Minerva-7B-instruct-v1.0/${filename}.out bash/_evaluation.sh "Minerva-7B-instruct-v1.0" "$test_set" "predictions/Minerva-7B-instruct-v1.0/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fM7BIV1-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_M7BIV1/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_M7BIV1 "$test_set" "predictions/LlamaDictionary-${language}_M7BIV1/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
done


## SWEDISH - SV ##
#__________________________________________________________________#
language="sv"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L38BI-evaluation --output=${slurm_folder}/$language/Llama-3-8B-instruct/${filename}.out bash/_evaluation.sh Llama-3-8B-instruct "$test_set" "predictions/Llama-3-8B-instruct/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L38BI "$test_set" "predictions/LlamaDictionary-${language}_L38BI/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "KB/bert-base-swedish-cased"
done


## NORWEGIAN - NO ##
#__________________________________________________________________#
language="no"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "NbAiLab/nb-bert-base"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "NbAiLab/nb-bert-base"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "NbAiLab/nb-bert-base"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "NbAiLab/nb-bert-base"
done


## SPANISH - ES ##
#__________________________________________________________________#
language="es"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dccuchile/bert-base-spanish-wwm-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dccuchile/bert-base-spanish-wwm-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dccuchile/bert-base-spanish-wwm-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dccuchile/bert-base-spanish-wwm-cased"
done


## JAPANESE - JA ##
#__________________________________________________________________#
language="ja"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "tohoku-nlp/bert-base-japanese"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "tohoku-nlp/bert-base-japanese"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "tohoku-nlp/bert-base-japanese"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "tohoku-nlp/bert-base-japanese"
done


## GERMAN - DE ##
#__________________________________________________________________#
language="de"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "google-bert/bert-base-german-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "google-bert/bert-base-german-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "google-bert/bert-base-german-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "google-bert/bert-base-german-cased"
done


## PORTUGUESE - PT ##
#__________________________________________________________________#
language="pt"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "neuralmind/bert-base-portuguese-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "neuralmind/bert-base-portuguese-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "neuralmind/bert-base-portuguese-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "neuralmind/bert-base-portuguese-cased"
done


## TURKISH - TR ##
#__________________________________________________________________#
language="tr"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dbmdz/bert-base-turkish-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dbmdz/bert-base-turkish-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dbmdz/bert-base-turkish-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dbmdz/bert-base-turkish-cased"
done


## MG ##
#__________________________________________________________________#
language="mg"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"
done


## DA ##
#__________________________________________________________________#
language="da"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "Maltehb/danish-bert-botxo"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "Maltehb/danish-bert-botxo"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "Maltehb/danish-bert-botxo"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "Maltehb/danish-bert-botxo"
done


## CA ##
#__________________________________________________________________#
language="ca"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "PlanTL-GOB-ES/roberta-base-ca"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "PlanTL-GOB-ES/roberta-base-ca"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "PlanTL-GOB-ES/roberta-base-ca"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "PlanTL-GOB-ES/roberta-base-ca"
done


## LT ##
#__________________________________________________________________#
language="lt"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "EMBEDDIA/litlat-bert"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "EMBEDDIA/litlat-bert"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "EMBEDDIA/litlat-bert"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "EMBEDDIA/litlat-bert"
done


## LA ##
#__________________________________________________________________#
language="la"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "ashleygong03/bamman-burns-latin-bert"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "ashleygong03/bamman-burns-latin-bert"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "ashleygong03/bamman-burns-latin-bert"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "ashleygong03/bamman-burns-latin-bert"
done


## PL ##
#__________________________________________________________________#
language="pl"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    #sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dkleczek/bert-base-polish-uncased-v1"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dkleczek/bert-base-polish-uncased-v1"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dkleczek/bert-base-polish-uncased-v1"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dkleczek/bert-base-polish-uncased-v1"
done


## KU ##
#__________________________________________________________________#
language="ku"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    #sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "asosoft/KuBERT-Central-Kurdish-BERT-Model"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "FacebookAI/xlm-roberta-base"
done


## EL ##
#__________________________________________________________________#
language="el"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "nlpaueb/bert-base-greek-uncased-v1"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "nlpaueb/bert-base-greek-uncased-v1"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "nlpaueb/bert-base-greek-uncased-v1"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "nlpaueb/bert-base-greek-uncased-v1"
done


## ZH ##
#__________________________________________________________________#
language="zh"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "google-bert/bert-base-chinese"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "google-bert/bert-base-chinese"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "google-bert/bert-base-chinese"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "google-bert/bert-base-chinese"
done


## FI ##
#__________________________________________________________________#
language="fi"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "TurkuNLP/bert-base-finnish-cased-v1"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "TurkuNLP/bert-base-finnish-cased-v1"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "TurkuNLP/bert-base-finnish-cased-v1"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "TurkuNLP/bert-base-finnish-cased-v1"
done


## RU ##
#__________________________________________________________________#
language="ru"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "DeepPavlov/rubert-base-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "DeepPavlov/rubert-base-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "DeepPavlov/rubert-base-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "DeepPavlov/rubert-base-cased"
done


## FRENCH - FR ##
#__________________________________________________________________#
language="fr"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dbmdz/bert-base-french-europeana-cased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dbmdz/bert-base-french-europeana-cased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dbmdz/bert-base-french-europeana-cased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dbmdz/bert-base-french-europeana-cased"
done


## ENGLISH - EN ##
#__________________________________________________________________#
language="en"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set")
    test_set="${test_set}_${language}.jsonl"

    echo "Filename: $filename"

    # Pre-trained
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "google-bert/bert-base-uncased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "google-bert/bert-base-uncased"

    # Fine-tuned
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "google-bert/bert-base-uncased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "google-bert/bert-base-uncased"
done



#### CROSS-LINGUAL evaluation using monolingual finetuned models
from_languages=("en" "de" "fr" "ru" "sv" "nl" "pl" "ku" "fi" "el" "ja" "ca" "es" "it" "tr" "pt" "zh" "no" "mg" "da" "lt" "la")
to_languages=("en" "fr" "de" "ru" "sv" "nl" "pl" "ku" "fi" "el" "ja" "ca" "es" "it" "tr" "pt" "zh" "no" "mg" "da" "lt" "la")

filename="train-dev-test/test_unseen_dbnary"

bert_models=("google-bert/bert-base-uncased" "dbmdz/bert-base-french-europeana-cased" "google-bert/bert-base-german-cased" "DeepPavlov/rubert-base-cased" "KB/bert-base-swedish-cased" "GroNLP/bert-base-dutch-cased" "dkleczek/bert-base-polish-uncased-v1" "FacebookAI/xlm-roberta-base" "TurkuNLP/bert-base-finnish-cased-v1" "nlpaueb/bert-base-greek-uncased-v1" "tohoku-nlp/bert-base-japanese" "PlanTL-GOB-ES/roberta-base-ca" "dccuchile/bert-base-spanish-wwm-cased" "dbmdz/bert-base-italian-uncased" "dbmdz/bert-base-turkish-cased" "neuralmind/bert-base-portuguese-cased" "google-bert/bert-base-chinese" "NbAiLab/nb-bert-base" "FacebookAI/xlm-roberta-base" "Maltehb/danish-bert-botxo" "EMBEDDIA/litlat-bert" "ashleygong03/bamman-burns-latin-bert")

# Loop through romance languages
for i in "${!to_languages[@]}"; do 
    lang1="${to_languages[$i]}"
    bert="${bert_models[$i]}"
    
    for lang2 in "${from_languages[@]}"; do
	if [[ -e "evaluation/LlamaDictionary-${lang2}_ML38BI/test_unseen_dbnary_${lang1}.tsv" ]]; then
	    continue
	fi

        if [[ "$lang1" != "$lang2" ]]; then
	    echo $lang1 $lang2 $bert
	    echo "${slurm_folder}/$lang1/LlamaDictionary-${lang2}_ML38BI/${filename}_${lang2}.out"

	    sbatch --time 6:00:00 --job-name ${lang1}_${lang2}-evaluation --output=${slurm_folder}/$lang1/LlamaDictionary-${lang2}_ML38BI/${filename}_${lang2}.out bash/_evaluation.sh LlamaDictionary-${lang2}_ML38BI "${filename}_${lang1}.jsonl" "predictions/LlamaDictionary-${lang2}_ML38BI/test_unseen_dbnary_${lang1}.txt" $lang1 "${bert}"
	    
        fi
	
    done
done
