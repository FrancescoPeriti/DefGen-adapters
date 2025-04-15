# Train-dev-test split

## DUTCH - NL ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_nl.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 66789       -- # Train targets: 47022      ###
### # Test unseen examples: 11155 -- # Test unseen targets: 6718 ###
### # Test seen examples: 11155   -- # Test seen targets: 9139   ###
### # Dev examples: 4401          -- # Dev targets: 2687         ###
### # tot examples: 93500         -- # tot targets: 53740        ###
#__________________________________________________________________#


## ITALIAN - IT ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_it.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 12978 -- # Train targets: 7938             ###
### # Test unseen examples: 2266 -- # Test unseen targets: 1134  ###
### # Test seen examples: 2266 -- # Test seen targets: 1488      ###
### # Dev examples: 877 -- # Dev targets: 453                    ###
### # tot examples: 18387 -- # tot targets: 9072                 ###
#__________________________________________________________________#


## SWEDISH - SV ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_sv.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 92063 -- # Train targets: 46860            ###
### # Test unseen examples: 15109 -- # Test unseen targets: 6695 ###
### # Test seen examples: 15109 -- # Test seen targets: 9840     ###
### # Dev examples: 6066 -- # Dev targets: 2677                  ###
### # tot examples: 128347 -- # tot targets: 53555               ###
#__________________________________________________________________#


## NORWEGIAN - NO ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_no.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 4521 -- # Train targets: 2775              ###
### # Test unseen examples: 715 -- # Test unseen targets: 397    ###
### # Test seen examples: 715 -- # Test seen targets: 483        ###
### # Dev examples: 313 -- # Dev targets: 158                    ###
### # tot examples: 6264 -- # tot targets: 3172                  ###
#__________________________________________________________________#


## JAPANESE - JA ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_no.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 17459 -- # Train targets: 8057             ###
### # Test unseen examples: 2751 -- # Test unseen targets: 1152  ###
### # Test seen examples: 2751 -- # Test seen targets: 1342      ###
### # Dev examples: 1277 -- # Dev targets: 460                   ###
### # tot examples: 24238 -- # tot targets: 9209                 ###
#__________________________________________________________________#


## SPANISH - ES ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_es.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 13820 -- # Train targets: 9308             ###
### # Test unseen examples: 2343 -- # Test unseen targets: 1330  ###
### # Test seen examples: 2343 -- # Test seen targets: 1436      ###
### # Dev examples: 870 -- # Dev targets: 532                    ###
### # tot examples: 19376 -- # tot targets: 10638                ###
#__________________________________________________________________#


## GERMAN - DE ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_de.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 284720 -- # Train targets: 117786          ###
### # Test unseen examples: 47663 -- # Test unseen targets: 16827###
### # Test seen examples: 47663 -- # Test seen targets: 27326    ###
### # Dev examples: 19220 -- # Dev targets: 6730                 ###
### # tot examples: 399266 -- # tot targets: 134613              ###
#__________________________________________________________________#


## RUSSIAN - RU ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_ru.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 156259 -- # Train targets: 101847           ###
### # Test unseen examples: 26197 -- # Test unseen targets: 14550 ###
### # Test seen examples: 26197 -- # Test seen targets: 15165     ###
### # Dev examples: 10486 -- # Dev targets: 5820                  ###
### # tot examples: 219139 -- # tot targets: 116397               ###
#__________________________________________________________________#


## PORTOGUESE - PT ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_pt.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 10010 -- # Train targets: 6650             ###
### # Test unseen examples: 1573 -- # Test unseen targets: 950   ###
### # Test seen examples: 1573 -- # Test seen targets: 908       ###
### # Dev examples: 663 -- # Dev targets: 380                    ###
### # tot examples: 13819 -- # tot targets: 7600                 ###
#__________________________________________________________________#


## GREEK - EL ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_el.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 20743 -- # Train targets: 15010            ###
### # Test unseen examples: 3473 -- # Test unseen targets: 2145  ###
### # Test seen examples: 3473 -- # Test seen targets: 2368      ###
### # Dev examples: 1397 -- # Dev targets: 857                   ###
### # tot examples: 29086 -- # tot targets: 17155                ###
#__________________________________________________________________#


## FRENCH - FR ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_fr.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 428152 -- # Train targets: 189854           ###
### # Test unseen examples: 71687 -- # Test unseen targets: 27122 ###
### # Test seen examples: 71687 -- # Test seen targets: 43334     ###
### # Dev examples: 28569 -- # Dev targets: 10848                 ###
### # tot examples: 600095 -- # tot targets: 216976               ###
#__________________________________________________________________#


## TURKISH - TR ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_tr.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 14217 -- # Train targets: 8883             ###
### # Test unseen examples: 2110 -- # Test unseen targets: 1270  ###
### # Test seen examples: 774 -- # Test seen targets: 618        ###
### # Dev examples: 835 -- # Dev targets: 507                    ###
### # tot examples: 17936 -- # tot targets: 10153                ###
#__________________________________________________________________#


## MG ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_mg.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 5684 -- # Train targets: 5670              ###
### # Test unseen examples: 816 -- # Test unseen targets: 811    ###
### # Test seen examples: 0 -- # Test seen targets: 0            ###
### # Dev examples: 325 -- # Dev targets: 324                    ###
### # tot examples: 6825 -- # tot targets: 6481                  ###
#__________________________________________________________________#


## DA ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_da.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 517 -- # Train targets: 392                ###
### # Test unseen examples: 82 -- # Test unseen targets: 56      ###
### # Test seen examples: 73 -- # Test seen targets: 49          ###
### # Dev examples: 35 -- # Dev targets: 22                      ###
### # tot examples: 707 -- # tot targets: 448                    ###
#__________________________________________________________________#


## CA ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_ca.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 20074 -- # Train targets: 13937            ###
### # Test unseen examples: 2908 -- # Test unseen targets: 1992  ###
### # Test seen examples: 646 -- # Test seen targets: 494        ###
### # Dev examples: 1259 -- # Dev targets: 796                   ###
### # tot examples: 24887 -- # tot targets: 15929                ###
#__________________________________________________________________#


## LT ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_lt.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 168 -- # Train targets: 99                 ###
### # Test unseen examples: 50 -- # Test unseen targets: 15      ###
### # Test seen examples: 26 -- # Test seen targets: 10          ###
### # Dev examples: 6 -- # Dev targets: 5                        ###
### # tot examples: 250 -- # tot targets: 114                    ###
#__________________________________________________________________#


## LA ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_la.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 132 -- # Train targets: 98                 ###
### # Test unseen examples: 26 -- # Test unseen targets: 14      ###
### # Test seen examples: 23 -- # Test seen targets: 20          ###
### # Dev examples: 7 -- # Dev targets: 5                        ###
### # tot examples: 188 -- # tot targets: 112                    ###
#__________________________________________________________________#


## ID ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_id.jsonl --output_folder train-dev-test
#__________________________________________________________________#
#NO example sentences available (only one)
#__________________________________________________________________#


## PL ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_pl.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 50410 -- # Train targets: 38532            ###
### # Test unseen examples: 8158 -- # Test unseen targets: 5505  ###
### # Test seen examples: 6662 -- # Test seen targets: 5192      ###
### # Dev examples: 3248 -- # Dev targets: 2202                  ###
### # tot examples: 68478 -- # tot targets: 44037                ###
#__________________________________________________________________#


## KU ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_ku.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 38577 -- # Train targets: 31095            ###
### # Test unseen examples: 5988 -- # Test unseen targets: 4443  ###
### # Test seen examples: 3648 -- # Test seen targets: 2434      ###
### # Dev examples: 2408 -- # Dev targets: 1777                  ###
### # tot examples: 50621 -- # tot targets: 35538                ###
#__________________________________________________________________#


## ZH ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_zh.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 8420 -- # Train targets: 3797              ###
### # Test unseen examples: 1392 -- # Test unseen targets: 543   ###
### # Test seen examples: 1392 -- # Test seen targets: 726       ###
### # Dev examples: 638 -- # Dev targets: 217                    ###
### # tot examples: 11842 -- # tot targets: 4340                 ###
#__________________________________________________________________#


## FINNISH - FI ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_fi.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 24228 -- # Train targets: 15704            ###
### # Test unseen examples: 4028 -- # Test unseen targets: 2244  ###
### # Test seen examples: 4028 -- # Test seen targets: 2843      ###
### # Dev examples: 1547 -- # Dev targets: 897                   ###
### # tot examples: 33831 -- # tot targets: 17948                ###
#__________________________________________________________________#


## ENGLISH - EN ##
python src/train-dev-test.py --train_size 0.75 --test_size 0.20 --seed 42 --dbnary_filename data/dbnary_en.jsonl --output_folder train-dev-test
#__________________________________________________________________#
### # Train examples: 440865 -- # Train targets: 180060           ###
### # Test unseen examples: 73168 -- # Test unseen targets: 25723 ###
### # Test seen examples: 73168 -- # Test seen targets: 39906     ###
### # Dev examples: 29200 -- # Dev targets: 10289                 ###
### # tot examples: 616401 -- # tot targets: 205783               ###
#__________________________________________________________________#

