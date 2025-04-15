slurm_folder="slurm-output/generation"

test_sets="train-dev-test/test_unseen_dbnary_en.jsonl train-dev-test/test_unseen_dbnary_de.jsonl train-dev-test/test_unseen_dbnary_fr.jsonl train-dev-test/test_unseen_dbnary_ru.jsonl train-dev-test/test_unseen_dbnary_sv.jsonl train-dev-test/test_unseen_dbnary_nl.jsonl train-dev-test/test_unseen_dbnary_pl.jsonl train-dev-test/test_unseen_dbnary_ku.jsonl train-dev-test/test_unseen_dbnary_fi.jsonl train-dev-test/test_unseen_dbnary_el.jsonl train-dev-test/test_unseen_dbnary_ja.jsonl train-dev-test/test_unseen_dbnary_ca.jsonl train-dev-test/test_unseen_dbnary_es.jsonl train-dev-test/test_unseen_dbnary_it.jsonl train-dev-test/test_unseen_dbnary_tr.jsonl train-dev-test/test_unseen_dbnary_pt.jsonl train-dev-test/test_unseen_dbnary_zh.jsonl train-dev-test/test_unseen_dbnary_no.jsonl train-dev-test/test_unseen_dbnary_da.jsonl train-dev-test/test_unseen_dbnary_lt.jsonl train-dev-test/test_unseen_dbnary_la.jsonl train-dev-test/test_unseen_dbnary_mg.jsonl"

languages="it es fr"
sbatch --time 3-01:00:00 --job-name it-es-fr-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1236" "$test_sets" 35 "$languages"

languages="sv de en"
sbatch --time 3-01:00:00 --job-name sv-de-en-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1179" "$test_sets" 20 "$languages"

languages="it es sv de fr en"
sbatch --time 3-01:00:00 --job-name it-es-sv-de-fr-en-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1612" "$test_sets" 27 "$languages"

languages="it es tr ja el fi ku pl sv ru de fr en" # 13k
sbatch --time 3-01:00:00 --job-name all-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-7496" "$test_sets" 35 "$languages"

languages="ja fi ru"
sbatch --time 3-01:00:00 --job-name ja-fi-ru-gen --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-3408" "$test_sets" 25 "$languages"

languages="pl ru"
sbatch --time 3-01:00:00 --job-name pl-ru-gen --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1148" "$test_sets" 25 "$languages"

languages="it es pl ru fr"
sbatch --time 3-01:00:00 --job-name it-es-pl-ru-fr-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1400" "$test_sets" 25 "$languages"

languages="it es pl sv ru de fr en"
sbatch --time 3-01:00:00 --job-name i1t-es-pl-ru-fr-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-3279" "$test_sets" 25 "$languages"

languages="pl sv ru de en"
sbatch --time 3-01:00:00 --job-name pl-sv-ru-deen-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1360" "$test_sets" 25 "$languages"

languages="it es pl ru fr"
sbatch --time 3-18:00:00 --job-name pl-sv-ru-deen-generation --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/test_unseen_dbnary.out" bash/_generation.sh meta-llama/Meta-Llama-3-8B-Instruct "models/LlamaDictionary-${languages}_ML38BI/checkpoint-1400" "$test_sets" 25 "$languages"
