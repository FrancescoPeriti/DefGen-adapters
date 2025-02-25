test_sets=("train-dev-test/test_seen_dbnary_nl.jsonl" "train-dev-test/test_unseen_dbnary_nl.jsonl")

pretrained="False"
for test_set in "${test_sets[@]}"; do
    filename=$(basename "$test_set" .jsonl)
    echo "Filename: $filename"
	
    if [ "$pretrained" == "True" ]; then
	sbatch \
	    --time 18:00:00 \
	    --job-name BVG7BU-generation \
	    --output=slurm-output/generation/GEITje-7B-ultra/${filename}.out \
	    bash/_generation.sh \
	    "BramVanroy/GEITje-7B-ultra" \
	    "None" \
	    "$test_set" \
	    20

	#sbatch \
	#    --time 18:00:00 \
	#    --job-name CL27BI-generation \
	#    --output=slurm-output/generation/ChocoLlama-2-7B-instruct/${filename}.out \
	#    bash/_generation.sh \
	#    "ChocoLlama/ChocoLlama-2-7B-instruct" \
	#    "None" \
	#    "$test_set" \
	#    20
	
	#sbatch \
	#    --time 18:00:00 \
	#    --job-name L3CL8BI-generation \
	#    --output=slurm-output/generation/Llama-3-ChocoLlama-8B-instruct/${filename}.out \
	#    bash/_generation.sh \
	#    "ChocoLlama/Llama-3-ChocoLlama-8B-instruct" \
	#    "None" \
	#    "$test_set" \
	#    30
	
	sbatch \
	    --time 18:00:00 \
	    --job-name L27BCH-generation \
	    --output=slurm-output/generation/Llama-2-7b-chat-hf/${filename}.out \
	    bash/_generation.sh \
	    "meta-llama/Llama-2-7b-chat-hf" \
	    "None" \
	    "$test_set" \
	    30
	
	sbatch \
	    --time 18:00:00 \
	    --job-name ML38BI-generation \
	    --output=slurm-output/generation/Meta-Llama-3-8B-Instruct/${filename}.out \
	    bash/_generation.sh \
	    "meta-llama/Meta-Llama-3-8B-Instruct" \
	    "None" \
	    "$test_set" \
	    20
    else

	sbatch \
            --time 18:00:00 \
            --job-name fBVG7BU-generation \
            --output=slurm-output/generation/LlamaDictionary-nl_BVG7BU/${filename}.out \
            bash/_generation.sh \
            "BramVanroy/GEITje-7B-ultra" \
            "models/LlamaDictionary-nl_BVG7BU/checkpoint-19614" \
            "$test_set" \
            20

	continue
	#sbatch \
	#    --time 18:00:00 \
	#    --job-name fCL27BI-generation \
	#    --output=slurm-output/generation/LlamaDictionary-nl_CL27BI/${filename}.out \
	#    bash/_generation.sh \
	#    "ChocoLlama/ChocoLlama-2-7B-instruct" \
	#    "models/LlamaDictionary-nl_CL27BI/checkpoint-19152" \
	#    "$test_set" \
	#    20

	#sbatch \
	#    --time 18:00:00 \
	#    --job-name fL3CL8BI-generation \
	#    --output=slurm-output/generation/LlamaDictionary-nl_L3CL8BI/${filename}.out \
	#    bash/_generation.sh \
	#    "ChocoLlama/Llama-3-ChocoLlama-8B-instruct" \
	#    "models/LlamaDictionary-nl_L3CL8BI/checkpoint-20250" \
	#    "$test_set" \
	#    30

	sbatch \
	    --time 18:00:00 \
	    --job-name fL27BCH-generation \
	    --output=slurm-output/generation/LlamaDictionary-nl_L27BCH/${filename}.out \
	    bash/_generation.sh \
	    "meta-llama/Llama-2-7b-chat-hf" \
	    "models/LlamaDictionary-nl_L27BCH/checkpoint-25607" \
	    "$test_set" \
	    20

        sbatch \
	    --time 1-18:00:00 \
	    --job-name fML38BI-generation \
	    --output=slurm-output/generation/LlamaDictionary-nl_ML38BI/${filename}.out \
	    bash/_generation.sh \
	    "meta-llama/Meta-Llama-3-8B-Instruct" \
	    "models/LlamaDictionary-nl_ML38BI/checkpoint-25110" \
	    "$test_set" \
	    35
    fi
done

