# Evaluation

slurm_folder="slurm-output/evaluation"
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
    continue # Done
    
    echo "Filename: $filename"

    # Pre-trained
    #sbatch --time 6:00:00 --job-name BVG7BU-evaluation --output=${slurm_folder}/$language/GEITje-7B-ultra/${filename}.out bash/_evaluation.sh "GEITje-7B-ultra" "$test_set" "predictions/GEITje-7B-ultra/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name L27BCH-evaluation --output=${slurm_folder}/$language/Llama-2-7b-chat-hf/${filename}.out bash/_evaluation.sh Llama-2-7b-chat-hf "$test_set" "predictions/Llama-2-7b-chat-hf/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name ML38BI-evaluation --output=${slurm_folder}/$language/Meta-Llama-3-8B-Instruct/${filename}.out bash/_evaluation.sh Meta-Llama-3-8B-Instruct "$test_set" "predictions/Meta-Llama-3-8B-Instruct/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"

    # Fine-tuned
    #sbatch --time 6:00:00 --job-name fBVG7BU-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_BVG7BU/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_BVG7BU "$test_set" "predictions/LlamaDictionary-${language}_BVG7BU/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name fL27BCH-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_L27BCH/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_L27BCH "$test_set" "predictions/LlamaDictionary-${language}_L27BCH/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
    sbatch --time 6:00:00 --job-name fML38BI-evaluation --output=${slurm_folder}/$language/LlamaDictionary-${language}_ML38BI/${filename}.out bash/_evaluation.sh LlamaDictionary-${language}_ML38BI "$test_set" "predictions/LlamaDictionary-${language}_ML38BI/${filename}_${language}.txt" $language "dbmdz/bert-base-italian-uncased"
done
