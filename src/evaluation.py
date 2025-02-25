import string
import random
import evaluate
import argparse
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset
from collections import defaultdict
from nltk.translate import bleu_score, nist_score
from comet import download_model, load_from_checkpoint

def format_prediction(text, args):
    if 'ML38BI' in args.predictions or 'Meta-Llama-3-8B-Instruct' in args.predictions:
        text = text.split('assistant<|end_header_id|>')[1]
    elif 'L27BCH' in args.predictions or 'Llama-2-7b-chat-hf' in args.predictions:
        text = text.split('[/INST]')[1]
    elif 'GEITje-7B-ultra' in args.predictions:
        text = text.split('<|assistant|>')[1]
    elif 'BVG7BU' in args.predictions:
        text = text.split('<|assistant|>')[1].split('.')[0] + '.'
    return text.strip()

def load_data(args):
    test_dataset = load_dataset('json', data_files=args.test_set, split='train', trust_remote_code=True)
    predictions = [format_prediction(line.strip(), args) for line in open(args.predictions, mode='r', encoding='utf-8')]
    test_dataset = test_dataset.add_column("prediction", predictions)
    return test_dataset.to_pandas()

def evaluation(df, args):
    random.seed(args.seed)

    eval_ = {
        "rougeL": (evaluate.load("rouge"), "rougeL"),
        "meteor": (evaluate.load("meteor"), "meteor"),
        "bertscore": (evaluate.load("bertscore"), "f1"),
        "sacrebleu": (evaluate.load("sacrebleu"), "score"),
        "exact_match": (evaluate.load("exact_match"), "exact_match"),
    }

    results = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        for metric in args.metrics:
            def_, pred_ = row['definition'], row['prediction']

            if metric == "nltk_bleu":
                auto_reweigh = False if len(pred_.split()) == 0 else True
                results[metric].append(bleu_score.sentence_bleu([def_.split()], pred_.split(),
                                                                smoothing_function=bleu_score.SmoothingFunction().method2,
                                                                auto_reweigh=auto_reweigh))
            elif metric == "nist":
                n = 5
                pred_len = len(pred_.split())
                if pred_len < 5:
                    n = pred_len
                try:
                    results[metric].append(nist_score.sentence_nist([def_.split()], pred_.split(), n=n))
                except:
                    results[metric].append(0)

            elif metric == "bertscore":
                evaluator, output_key = eval_[metric]
                results[metric].append(evaluator.compute(predictions=[pred_],
                                                         references=[def_], lang=args.language,
                                                         model_type=args.model_type, num_layers=12)[output_key][0])
            else:
                evaluator, output_key = eval_[metric]
                results[metric].append(evaluator.compute(predictions=[pred_],
                                                         references=[def_])[output_key])
    return results

def xcomet_evaluation(df, args):
    login(args.hugginface_token)

    # xcomet
    model_path = download_model(args.xcomet_model)
    xcomet = load_from_checkpoint(model_path)

    data = list()
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        data.append(dict(ref=row['definition'], mt=row['prediction']))

    model_output = xcomet.predict(data, batch_size=16, gpus=1)
    df['xcomet_seg'] = model_output.scores
    df['xcomet_sys'] = [model_output.system_score]*len(data)
    df['xcomet_spans'] = model_output.metadata.error_spans
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcomet_model", type=str, default="Unbabel/XCOMET-XL")
    parser.add_argument("--hugginface_token", type=str, default="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")
    parser.add_argument("--output_folder", type=str, default='results')
    parser.add_argument("--metrics", nargs='*', type=str, default=["rougeL", "nltk_bleu", "nist", "sacrebleu", "meteor", "bertscore", "exact_match"])
    parser.add_argument("--test_set", type=str, default="train_dev_test/test.jsonl")
    parser.add_argument("--predictions", type=str, default="predictions/test.txt")
    parser.add_argument("--model_type", type=str, default="GroNLP/bert-base-dutch-cased")
    parser.add_argument("--language", type=str, default="nl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args)
    df = xcomet_evaluation(df, args)
    results = evaluation(df, args)

    for metric, values in results.items():
        df[metric] = values

    stem = Path(args.test_set).stem
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{args.output_folder}/{stem}.tsv', sep='\t', index=False)
