folder_name="/cephyr/users/stefanod/Alvis/input"

for file_path in "${folder_name}"/*.jsonl; do
    file_name=$(basename "${file_path}" .jsonl)
    echo $file_name
    sbatch \
	--time 1-18:00:00 \
	--job-name fML38BI-generation \
	--output="Stefano_slurm-output/generation/LlamaDictionary-nl_ML38BI/${file_name}.out" \
	bash/_generation.sh \
	"meta-llama/Meta-Llama-3-8B-Instruct" \
	"models/LlamaDictionary-nl_ML38BI/checkpoint-25110" \
	"${file_path}" \
	35
done
