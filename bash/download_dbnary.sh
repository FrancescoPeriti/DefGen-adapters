#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A100fat:1

# -C NOGPU

#apptainer build my_container.sif my_recipe.def

#wget -O dbnary_zh.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/zh_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_zh.ttl.bz2
#wget -O dbnary_tr.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/tr_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_tr.ttl.bz2
#wget -O dbnary_sv.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/sv_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_sv.ttl.bz2
#wget -O dbnary_sh.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/sh_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_sh.ttl.bz2
#wget -O dbnary_ru.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ru_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ru.ttl.bz2
#wget -O dbnary_pt.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/pt_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_pt.ttl.bz2
#wget -O dbnary_pl.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/pl_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_pl.ttl.bz2
#wget -O dbnary_no.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/no_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_no.ttl.bz2
#wget -O dbnary_nl.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/nl_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_nl.ttl.bz2
#wget -O dbnary_mg.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/mg_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_mg.ttl.bz2
#wget -O dbnary_lt.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/lt_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_lt.ttl.bz2
#wget -O dbnary_la.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/la_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_la.ttl.bz2
#wget -O dbnary_ku.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ku_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ku.ttl.bz2
#wget -O dbnary_ja.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ja_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ja.ttl.bz2
#wget -O dbnary_it.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/it_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_it.ttl.bz2
#wget -O dbnary_id.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/id_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_id.ttl.bz2
#wget -O dbnary_ga.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ga_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ga.ttl.bz2
#wget -O dbnary_fr.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/fr_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_fr.ttl.bz2
#wget -O dbnary_fi.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/fi_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_fi.ttl.bz2
#wget -O dbnary_es.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/es_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_es.ttl.bz2
#wget -O dbnary_en.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/en_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_en.ttl.bz2
#wget -O dbnary_el.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/el_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_el.ttl.bz2
#wget -O dbnary_de.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/de_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_de.ttl.bz2
#wget -O dbnary_da.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/da_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_da.ttl.bz2
#wget -O dbnary_ca.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ca_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ca.ttl.bz2
#wget -O dbnary_bg.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/bg_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_bg.ttl.bz2

#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_bg.ttl --output data/DBNARY/dbnary_bg.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_ca.ttl --output data/DBNARY/dbnary_ca.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_da.ttl --output data/DBNARY/dbnary_da.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_de.ttl --output data/DBNARY/dbnary_de.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_el.ttl --output data/DBNARY/dbnary_el.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_en.ttl --output data/DBNARY/dbnary_en.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_es.ttl --output data/DBNARY/dbnary_es.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_fi.ttl --output data/DBNARY/dbnary_fi.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_fr.ttl --output data/DBNARY/dbnary_fr.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_ga.ttl --output data/DBNARY/dbnary_ga.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_id.ttl --output data/DBNARY/dbnary_id.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_it.ttl --output data/DBNARY/dbnary_it.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_ja.ttl --output data/DBNARY/dbnary_ja.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_ku.ttl --output data/DBNARY/dbnary_ku.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_la.ttl --output data/DBNARY/dbnary_la.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_lt.ttl --output data/DBNARY/dbnary_lt.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_mg.ttl --output data/DBNARY/dbnary_mg.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_nl.ttl --output data/DBNARY/dbnary_nl.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_no.ttl --output data/DBNARY/dbnary_no.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_pl.ttl --output data/DBNARY/dbnary_pl.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_pt.ttl --output data/DBNARY/dbnary_pt.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_ru.ttl --output data/DBNARY/dbnary_ru.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_sh.ttl --output data/DBNARY/dbnary_sh.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_sv.ttl --output data/DBNARY/dbnary_sv.jsonl
#apptainer exec my_container.sif python src/process_dbnary.py --filename dbnary_tr.ttl --output data/DBNARY/dbnary_tr.jsonl
apptainer exec --nv my_container.sif python src/process_dbnary.py --filename dbnary_zh.ttl --output data/DBNARY/dbnary_zh.jsonl



#apptainer exec --nv my_container.sif python src/train_dev_test_split.py
apptainer exec --nv my_container.sif python src/train_dev_test_split.py --languages en it nl
python src/process_split.py --languages "en" "it" "nl"

apptainer exec --nv my_container.sif python src/train.py --base_model_name "Unbabel/TowerInstruct-7B-v0.2" --hugginface_token "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" --train_filename "data/train.jsonl" --cache_dir "/mimer/NOBACKUP/groups/cik_data/fra_hf_cache" --seed 42 --finetuned_model_name "TowerLanguageBridge-7B" --output_dir "models" --max_seq_length 2048 --verbose --weight_decay 0.01 --warmup_ratio 0.05 --batch_size 64 --learning_rate 7e-06 --num_train_epochs 5 --gradient_accumulation_steps 1 --mask_prob 0.15


