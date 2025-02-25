test_sets=("train-dev-test/test_seen_dbnary_nl.jsonl" "train-dev-test/test_unseen_dbnary_nl.jsonl")

pretrained="False"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set" .jsonl)
    echo "Filename: $filename"

    if [ "$pretrained" == "True" ]; then
	sbatch \
            --time 6:00:00 \
            --job-name BVG7BU-evaluation \
            --output=slurm-output/evaluation/GEITje-7B-ultra/${filename}.out \
            bash/_evaluation.sh \
            "GEITje-7B-ultra" \
	    "$test_set" \
	    "predictions/GEITje-7B-ultra/${filename}.txt"

        #sbatch \
        #    --time 6:00:00 \
        #    --job-name CL27BI-evaluation \
        #    --output=slurm-output/evaluation/ChocoLlama-2-7B-instruct/${filename}.out \
        #    bash/_evaluation.sh \
        #    "ChocoLlama-2-7B-instruct" \
	#    "$test_set" \
	#    "predictions/ChocoLlama-2-7B-instruct/${filename}.txt"
	
        #sbatch \
        #    --time 6:00:00 \
        #    --job-name L3CL8BI-generation \
        #    --output=slurm-output/generation/Llama-3-ChocoLlama-8B-instruct/${filename}.out \
        #    bash/_generation.sh \
        #    "Llama-3-ChocoLlama-8B-instruct" \
	#    "$test_set" \
	#    "predictions/Llama-3-ChocoLlama-8B-instruct/${filename}.txt"
	
        sbatch \
            --time 6:00:00 \
            --job-name L27BCH-evaluation \
            --output=slurm-output/evaluation/Llama-2-7b-chat-hf/${filename}.out \
            bash/_evaluation.sh \
            "Llama-2-7b-chat-hf" \
	    "$test_set" \
	    "predictions/Llama-2-7b-chat-hf/${filename}.txt"

        sbatch \
            --time 6:00:00 \
            --job-name ML38BI-evaluation \
            --output=slurm-output/evaluation/Meta-Llama-3-8B-Instruct/${filename}.out \
            bash/_evaluation.sh \
            "Meta-Llama-3-8B-Instruct" \
	    "$test_set" \
	    "predictions/Meta-Llama-3-8B-Instruct/${filename}.txt"
    else

	sbatch \
            --time 6:00:00 \
            --job-name fBVG7BU-evaluation \
            --output=slurm-output/evaluation/LlamaDictionary-nl_BVG7BU/${filename}.out \
            bash/_evaluation.sh \
            "LlamaDictionary-nl_BVG7BU" \
	    "$test_set" \
	    "predictions/LlamaDictionary-nl_BVG7BU/${filename}.txt"
	continue
	
	sbatch \
            --time 6:00:00 \
            --job-name fCL27BI-evaluation \
            --output=slurm-output/evaluation/LlamaDictionary-nl_CL27BI/${filename}.out \
            bash/_evaluation.sh \
            "LlamaDictionary-nl_CL27BI" \
	    "$test_set" \
	    "predictions/LlamaDictionary-nl_CL27BI/${filename}.txt"
	
	sbatch \
            --time 6:00:00 \
            --job-name fL3CL8BI-evaluation \
            --output=slurm-output/evaluation/LlamaDictionary-nl_L3CL8BI/${filename}.out \
            bash/_evaluation.sh \
            "LlamaDictionary-nl_L3CL8BI" \
	    "$test_set" \
	    "predictions/LlamaDictionary-nl_L3CL8BI/${filename}.txt"

	
	sbatch \
            --time 6:00:00 \
            --job-name fL27BCH-evaluation \
            --output=slurm-output/evaluation/LlamaDictionary-nl_L27BCH/${filename}.out \
            bash/_evaluation.sh \
            "LlamaDictionary-nl_L27BCH" \
            "$test_set" \
	    "predictions/LlamaDictionary-nl_L27BCH/${filename}.txt"
	
        sbatch \
            --time 6:00:00 \
            --job-name fML38BI-evaluation \
            --output=slurm-output/evaluation/LlamaDictionary-nl_ML38BI/${filename}.out \
            bash/_evaluation.sh \
            "LlamaDictionary-nl_ML38BI" \
            "$test_set" \
	    "predictions/LlamaDictionary-nl_ML38BI/${filename}.txt"
    fi
done
