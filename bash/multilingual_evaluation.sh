slurm_folder="slurm-output/evaluation"

langs=("en" "fr" "de" "ru" "sv" "nl" "pl" "ku" "fi" "el" "ja" "ca" "es" "it" "tr" "pt" "zh" "no" "da" "lt" "la" "mg")
bert_models=("google-bert/bert-base-uncased" "dbmdz/bert-base-french-europeana-cased" "google-bert/bert-base-german-cased" "DeepPavlov/rubert-base-cased" "KB/bert-base-swedish-cased" "GroNLP/bert-base-dutch-cased" "dkleczek/bert-base-polish-uncased-v1" "FacebookAI/xlm-roberta-base" "TurkuNLP/bert-base-finnish-cased-v1" "nlpaueb/bert-base-greek-uncased-v1" "tohoku-nlp/bert-base-japanese" "PlanTL-GOB-ES/roberta-base-ca" "dccuchile/bert-base-spanish-wwm-cased" "dbmdz/bert-base-italian-uncased" "dbmdz/bert-base-turkish-cased" "neuralmind/bert-base-portuguese-cased" "google-bert/bert-base-chinese" "NbAiLab/nb-bert-base" "Maltehb/danish-bert-botxo" "EMBEDDIA/litlat-bert" "ashleygong03/bamman-burns-latin-bert" "FacebookAI/xlm-roberta-base")
test_sets=("train-dev-test/test_unseen_dbnary_en.jsonl" "train-dev-test/test_unseen_dbnary_fr.jsonl" "train-dev-test/test_unseen_dbnary_de.jsonl" "train-dev-test/test_unseen_dbnary_ru.jsonl" "train-dev-test/test_unseen_dbnary_sv.jsonl" "train-dev-test/test_unseen_dbnary_nl.jsonl" "train-dev-test/test_unseen_dbnary_pl.jsonl" "train-dev-test/test_unseen_dbnary_ku.jsonl" "train-dev-test/test_unseen_dbnary_fi.jsonl" "train-dev-test/test_unseen_dbnary_el.jsonl" "train-dev-test/test_unseen_dbnary_ja.jsonl" "train-dev-test/test_unseen_dbnary_ca.jsonl" "train-dev-test/test_unseen_dbnary_es.jsonl" "train-dev-test/test_unseen_dbnary_it.jsonl" "train-dev-test/test_unseen_dbnary_tr.jsonl" "train-dev-test/test_unseen_dbnary_pt.jsonl" "train-dev-test/test_unseen_dbnary_zh.jsonl" "train-dev-test/test_unseen_dbnary_no.jsonl" "train-dev-test/test_unseen_dbnary_da.jsonl" "train-dev-test/test_unseen_dbnary_lt.jsonl" "train-dev-test/test_unseen_dbnary_la.jsonl" "train-dev-test/test_unseen_dbnary_mg.jsonl")

#langs=("ku" "mg")
#bert_models=("FacebookAI/xlm-roberta-base" "FacebookAI/xlm-roberta-base")
#test_sets=("train-dev-test/test_unseen_dbnary_ku.jsonl" "train-dev-test/test_unseen_dbnary_mg.jsonl")

count=0
max_jobs=50
multilingual_models=("it es fr" "it es pl ru fr" "it es pl sv ru de fr en" "it es sv de fr en" "it es tr ja el fi ku pl sv ru de fr en" "pl ru" "pl sv ru de en" "sv de en" "ja fi ru")
for languages in "${multilingual_models[@]}"; do
    for i in "${!langs[@]}"; do
	lang="${langs[$i]}"


	test_set="${test_sets[$i]}"
	bert_model="${bert_models[$i]}"
	
	filename="${test_set//train-dev-test\//}"  # Remove 'train-dev-test/'
	filename="${filename%.jsonl}"  # Remove the '.jsonl' extension
	
	if [[ -e "evaluation/LlamaDictionary-${languages}_ML38BI/${filename}.tsv" ]]; then
            continue
	fi
	
	if [[ ! -e "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" ]]; then
	    echo "Missing: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
	    continue
	fi

	((count++))
	
	echo "$filename - $bert_model"
	#echo "Progress: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
	echo "${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out"

	#if [[ $count -le 30 ]]; then
	#continue
	#fi

	sbatch --time 6:00:00 --job-name "${lang}-evaluation" --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out" bash/_evaluation.sh "LlamaDictionary-${languages}_ML38BI" "$test_set" "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" $lang $bert_model
	
        #if [[ $count -ge $max_jobs ]]; then
        #    echo "Submitted $max_jobs jobs. Exiting."
        #    exit 0
	#fi
    done
done

exit


languages="it es fr" # la pt ca (same family unseen)
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    test_set="${test_sets[$i]}"
    bert_model="${bert_models[$i]}"
    
    filename="${test_set//train-dev-test\//}"  # Remove 'train-dev-test/'
    filename="${filename%.jsonl}"  # Remove the '.jsonl' extension

    if [[ -e "evaluation/LlamaDictionary-${languages}_ML38BI/${filename}.tsv" ]]; then
        continue
    fi

    if [[ ! -e "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" ]]; then
	echo "Missing: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
	continue
    fi
        
    echo "$filename - $bert_model"
    echo "Progress: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    sbatch --time 6:00:00 --job-name "${lang}-evaluation" --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out" bash/_evaluation.sh "LlamaDictionary-${languages}_ML38BI" "$test_set" "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" $lang $bert_model
done


languages="sv de en" # da no nl (same family unseen)
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    test_set="${test_sets[$i]}"
    bert_model="${bert_models[$i]}"
    
    filename="${test_set//train-dev-test\//}"  # Remove 'train-dev-test/'
    filename="${filename%.jsonl}"  # Remove the '.jsonl' extension

    if [[ -e "evaluation/LlamaDictionary-${languages}_ML38BI/${filename}.tsv" ]]; then
        continue
    fi

    if [[ ! -e "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" ]]; then
	echo "Missing: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    fi
    
    echo "$filename - $bert_model"
    echo "Progress: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    #echo "${lang}-evaluation"
    #echo "${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out"
    #echo "LlamaDictionary-${languages}_ML38BI"
    #echo "$test_set"
    #echo "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    sbatch --time 6:00:00 --job-name "${lang}-evaluation" --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out" bash/_evaluation.sh "LlamaDictionary-${languages}_ML38BI" "$test_set" "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" $lang $bert_model
done


languages="it es sv de fr en" # la pt ca da no nl
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    test_set="${test_sets[$i]}"
    bert_model="${bert_models[$i]}"
    
    filename="${test_set//train-dev-test\//}"  # Remove 'train-dev-test/'
    filename="${filename%.jsonl}"  # Remove the '.jsonl' extension
    
    if [[ -e "evaluation/LlamaDictionary-${languages}_ML38BI/${filename}.tsv" ]]; then
        continue
    fi
    
    if [[ ! -e "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" ]]; then
	echo "Missing: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    fi
    
    echo "$filename - $bert_model"
    echo "Progress: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    sbatch --time 6:00:00 --job-name "${lang}-evaluation" --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out" bash/_evaluation.sh "LlamaDictionary-${languages}_ML38BI" "$test_set" "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" $lang $bert_model
done



languages="it es tr ja el fi ku pl sv ru de fr en"
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    test_set="${test_sets[$i]}"
    bert_model="${bert_models[$i]}"
    
    filename="${test_set//train-dev-test\//}"  # Remove 'train-dev-test/'
    filename="${filename%.jsonl}"  # Remove the '.jsonl' extension
    
    if [[ -e "evaluation/LlamaDictionary-${languages}_ML38BI/${filename}.tsv" ]]; then
        continue
    fi

    if [[ ! -e "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" ]]; then
	echo "Missing: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    fi
    
    echo "$filename - $bert_model"
    echo "Progress: predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt"
    sbatch --time 6:00:00 --job-name "${lang}-evaluation" --output="${slurm_folder}/${languages}/LlamaDictionary-${languages}_ML38BI/${filename}.out" bash/_evaluation.sh "LlamaDictionary-${languages}_ML38BI" "$test_set" "predictions/LlamaDictionary-${languages}_ML38BI/${filename}.txt" $lang $bert_model
done
