#!/bin/bash
#SBATCH -A NAISS2024-5-148 -p alvis
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=A100fat:1

############################### -- Download -- ###############################
mkdir data
cd data

# Download chinese
wget -O dbnary_zh.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/zh_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_zh.ttl.bz2

# Download turkish
wget -O dbnary_tr.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/tr_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_tr.ttl.bz2

# Download swedish
wget -O dbnary_sv.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/sv_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_sv.ttl.bz2

# Download serbo-croatian
wget -O dbnary_sh.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/sh_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_sh.ttl.bz2

# Download russian
wget -O dbnary_ru.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ru_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ru.ttl.bz2

# Download portuguese
wget -O dbnary_pt.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/pt_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_pt.ttl.bz2

# Download polish
wget -O dbnary_pl.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/pl_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_pl.ttl.bz2

# Download norway
wget -O dbnary_no.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/no_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_no.ttl.bz2

# Download dutch
wget -O dbnary_nl.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/nl_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_nl.ttl.bz2

# Download malagasy
wget -O dbnary_mg.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/mg_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_mg.ttl.bz2

# Download lithuanian
wget -O dbnary_lt.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/lt_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_lt.ttl.bz2

# Download latin
wget -O dbnary_la.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/la_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_la.ttl.bz2

# Download kurdish
wget -O dbnary_ku.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ku_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ku.ttl.bz2

# Download japanese
wget -O dbnary_ja.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ja_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ja.ttl.bz2

# Download italian
wget -O dbnary_it.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/it_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_it.ttl.bz2

# Download ???
wget -O dbnary_id.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/id_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_id.ttl.bz2

# Download irish
wget -O dbnary_ga.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ga_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ga.ttl.bz2

# Download french
wget -O dbnary_fr.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/fr_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_fr.ttl.bz2

# Download finnish
wget -O dbnary_fi.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/fi_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_fi.ttl.bz2

# Download spanish
wget -O dbnary_es.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/es_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_es.ttl.bz2

# Download english
wget -O dbnary_en.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/en_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_en.ttl.bz2

# Download greek
wget -O dbnary_el.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/el_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_el.ttl.bz2

# Download german
wget -O dbnary_de.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/de_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_de.ttl.bz2

# Download danish
wget -O dbnary_da.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/da_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_da.ttl.bz2

# Download catalan
wget -O dbnary_ca.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/ca_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_ca.ttl.bz2

# Download Bulgarian
wget -O dbnary_bg.ttl.bz2 https://kaiko.getalp.org/static/ontolex/latest/bg_dbnary_ontolex.ttl.bz2; bzip2 -d dbnary_bg.ttl.bz2


############################### -- Process -- ###############################
cd ..

python src/process_dbnary.py --filename dbnary_bg.ttl --output data/dbnary_bg.jsonl
python src/process_dbnary.py --filename dbnary_ca.ttl --output data/dbnary_ca.jsonl
python src/process_dbnary.py --filename dbnary_da.ttl --output data/dbnary_da.jsonl
python src/process_dbnary.py --filename dbnary_de.ttl --output data/dbnary_de.jsonl
python src/process_dbnary.py --filename dbnary_el.ttl --output data/dbnary_el.jsonl
python src/process_dbnary.py --filename dbnary_en.ttl --output data/dbnary_en.jsonl
python src/process_dbnary.py --filename dbnary_es.ttl --output data/dbnary_es.jsonl
python src/process_dbnary.py --filename dbnary_fi.ttl --output data/dbnary_fi.jsonl
python src/process_dbnary.py --filename dbnary_fr.ttl --output data/dbnary_fr.jsonl
python src/process_dbnary.py --filename dbnary_ga.ttl --output data/dbnary_ga.jsonl
python src/process_dbnary.py --filename dbnary_id.ttl --output data/dbnary_id.jsonl
python src/process_dbnary.py --filename dbnary_it.ttl --output data/dbnary_it.jsonl
python src/process_dbnary.py --filename dbnary_ja.ttl --output data/dbnary_ja.jsonl
python src/process_dbnary.py --filename dbnary_ku.ttl --output data/dbnary_ku.jsonl
python src/process_dbnary.py --filename dbnary_la.ttl --output data/dbnary_la.jsonl
python src/process_dbnary.py --filename dbnary_lt.ttl --output data/dbnary_lt.jsonl
python src/process_dbnary.py --filename dbnary_mg.ttl --output data/dbnary_mg.jsonl
python src/process_dbnary.py --filename dbnary_nl.ttl --output data/dbnary_nl.jsonl
python src/process_dbnary.py --filename dbnary_no.ttl --output data/dbnary_no.jsonl
apython src/process_dbnary.py --filename dbnary_pl.ttl --output data/dbnary_pl.jsonl
python src/process_dbnary.py --filename dbnary_pt.ttl --output data/dbnary_pt.jsonl
python src/process_dbnary.py --filename dbnary_ru.ttl --output data/dbnary_ru.jsonl
python src/process_dbnary.py --filename dbnary_sh.ttl --output data/dbnary_sh.jsonl
python src/process_dbnary.py --filename dbnary_sv.ttl --output data/dbnary_sv.jsonl
python src/process_dbnary.py --filename dbnary_tr.ttl --output data/dbnary_tr.jsonl
python src/process_dbnary.py --filename dbnary_zh.ttl --output data/dbnary_zh.jsonl
