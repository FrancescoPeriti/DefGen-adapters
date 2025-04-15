# Definition Generation for Word Meaning Modeling: Monolingual, Multilingual, and Cross-Lingual Perspectives
This is the official repository for our paper _TITLE_

## Table of Contents

- [Abstract](#abstract)
- [Getting Started](#getting-started)
- [Reproducing Results](#reproducing-results)
- [References](#references)

## Abstract
bla bla bla ...

## Getting Started
Before you begin, ensure you have met the following requirements:
- <img src="https://static.wikia.nocookie.net/logopedia/images/1/1f/Nvidia_CUDA.svg/revision/latest?cb=20230319014140" width="40" height="25"> Cuda 12.4
- <img src="https://miro.medium.com/v2/resize:fit:1400/1*lSTuwS4exV_s__kcShxk8w.png" width="20" height="20"> Python 3.11.3
- <img src="https://cdn-images-1.medium.com/max/580/0*Kt5_0uGLlCFAgbt6.png" width="25" height="25"> Python packages (listed in `requirements.txt`)

If you are using a cluster, you can direcyly load Python 3.11 with:

```module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1```

To install the required packages, you can create a virtual environment and use pip:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Results
You find our bash scripts in the ```bash``` folder and our Python scripts in the ```src``` folder. Feel free to contact us if you face any issues!

### Download, process, and split data
To use our exact data, you can refer to our <img src="https://github.com/user-attachments/assets/08500240-788b-44ce-8f57-09edf410fb8a" width="18" height="18"> Hugging Face <a href="https://huggingface.co/collections/FrancescoPeriti/definitiongeneration-datasets-67fe0c84a2045316f04899d2">collection</a>.
Otherwise, follow the instructions below.

- Download and process <a href="https://kaiko.getalp.org/about-dbnary/">Dbnary</a> data. Our download is updated as of 25/02/2025.
```bash 
bash bash/download_dbnary.sh
```

- Split the data into Train, Dev, and Test partitions.
```bash 
bash bash/train-dev-test.sh
```

### Fine-tuning with LoRA
To use our models, you can refer to our <img src="https://huggingface.co/collections/FrancescoPeriti/definitiongeneration-models-67fdff7cf67464c769dd95b3" width="18" height="18"> Hugging Face <a href="https://huggingface.co/collections/FrancescoPeriti/definitiongeneration-datasets-67fe0c84a2045316f04899d2">collection</a>.
Otherwise, follow the instructions below.

- Fine-tune <a href="https://www.llama.com/">Llama</a> models. 
```bash 
bash bash/finetuning.sh # monolingual finetuning
bash bash/multilingual_finetuning.sh # multilingual finetuning
```

## Definition generation and evaluation
To use our models for Definition Generation, you can refer to our <img src="https://huggingface.co/collections/FrancescoPeriti/definitiongeneration-models-67fdff7cf67464c769dd95b3" width="18" height="18"> Hugging Face <a href="https://huggingface.co/collections/FrancescoPeriti/definitiongeneration-datasets-67fe0c84a2045316f04899d2">collection</a>.
Otherwise, follow the instructions below.

- Generate definitions with fine-tuned models.
```bash 
bash bash/generation.sh # definition generation with monolingual models
bash bash/multilingual_generation.sh # definition generation with multilingual models
```

To access our evaluation results, please refer to our <a href="TODO">Zenodo</a> page. Otherwise, follow the instructions below.
- Evaluate generated definitions.
```bash 
bash bash/evaluation.sh # definition generation with monolingual models
bash bash/multilingual_evaluation.sh # definition generation with multilingual models
```

## Plots
You can use the ```plot.ipynb``` notebook to generate all the plots.

## Reference
bla bla bla ...
