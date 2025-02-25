import re
import json
import string
import pathlib
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)

def fix_lemma_options(text):
    if text.endswith('}.'):
        text=text[:-1]+'#.'

    # Handle patterns like #{option1|option2}#
    text = re.sub(r"#\{[^|}]+\|([^}]+)\}#", r"\1", text)
    # Handle patterns like {option1|option2}# at the start of the sentence
    text = re.sub(r"^\{[^|}]+\|([^}]+)\}#", r"\1", text)
    return text

def fix_parentheses(text):
    if text.startswith("("):
        return text

    # Check if there is a pattern where a closing parenthesis is followed by any text and then an opening parenthesis
    if re.search(r"\)[^(]*\(", text):
        return "(" + text  # Add an opening parenthesis at the start

    # If the text contains a closing parenthesis before an opening one, prepend an opening parenthesis
    if ")" in text and "(" not in text:
        return "(" + text  # Add an opening parenthesis at the start

    return text

def clean(text):
    # remove \n characters
    text = text.replace('\n', '; ').strip()

    # remove punctuation as a last character
    if len(text) > 1 and text[-1] in string.punctuation:
        text = text[:-1]

    # remove double spaces
    text = " ".join(text.split()).strip()

    # remove punctuation as a first character
    if len(text) > 1 and text[0] in string.punctuation:
        text = text[1:]

    # uppercase first letter
    if len(text) > 1:
        text = text[0].upper() + text[1:]

    return fix_lemma_options(fix_parentheses(text))


def train_dev_test_split(df, args):
    targets = df['target'].unique()

    train_targets, remaining_targets = train_test_split(targets, test_size=1-args.train_size, random_state=args.seed)
    dev_targets, test_targets = train_test_split(remaining_targets, test_size=args.test_size / (1-args.train_size), random_state=args.seed)
    test_seen_targets, test_unseen_targets = train_test_split(remaining_targets, test_size=0.5, random_state=args.seed)
        
    train_df = df[df['target'].isin(train_targets)]
    test_unseen_df = df[df['target'].isin(test_unseen_targets)].sample(frac=1).reset_index(drop=True)
    dev_df = df[df['target'].isin(dev_targets)].sample(frac=1).reset_index(drop=True)

    # dealing with 'test seen'
    tmp_df = df[df['target'].isin(test_seen_targets)].sample(frac=1).reset_index(drop=True)
    train_df = pd.concat([tmp_df, train_df]).sample(frac=1).reset_index(drop=True)

    labels = list()
    flag = list()
    for i, row in train_df.iterrows():
        label = row['target']+row['definition']
        if label in labels:
            flag.append(1)
        else:
            labels.append(label)
            flag.append(0)

    train_df['flag'] = flag
    tmp_df = train_df[train_df['flag']==1].reset_index(drop=True)
    
    test_seen_df = tmp_df.iloc[:test_unseen_df.shape[0]].sample(frac=1).reset_index(drop=True)
    train_df = pd.concat([train_df[train_df['flag']==0], tmp_df.iloc[test_unseen_df.shape[0]:]])
    
    print(f'- # Train examples: {train_df.shape[0]} -- # Train targets: {len(train_df.target.unique())}')
    print(f'- # Test unseen examples: {test_unseen_df.shape[0]} -- # Test unseen targets: {len(test_unseen_df.target.unique())}')
    print(f'- # Test seen examples: {test_seen_df.shape[0]} -- # Test seen targets: {len(test_seen_df.target.unique())}')
    print(f'- # Dev examples: {dev_df.shape[0]} -- # Dev targets: {len(dev_targets)}')
    print(f'- # tot examples: {train_df.shape[0] + test_unseen_df.shape[0] + test_seen_df.shape[0] + dev_df.shape[0]} -- # tot targets: {len(targets)}')

    return train_df, dev_df, test_seen_df, test_unseen_df


def load_data(args, filename):
    records = list()
    for record in open(filename, mode='r', encoding='utf-8'):
        record = json.loads(record)
        if not len(record['examples']): continue

        record['target'] = clean(record['target'])
        record['definition'] = clean(record['definition'])

        if not len(record['target']) or record['target'] is None: continue
        if not len(record['definition']) or record['definition'] is None: continue
        if record['definition'][-1] != '.':
            record['definition']+='.'

        for example in record['examples']:
            example = clean(example)
            if not len(example) or example is None: continue
            records.append(dict(target=record['target'], definition=record['definition'], example=example))

            for variant in record['variants']:
                variant = clean(variant)
                if not len(variant) or variant is None: continue
                if example.count(record['target']) == 1:
                    records.append(dict(target=variant, definition=record['definition'], example=example.replace(record['target'], variant)))

    return pd.DataFrame(records).drop_duplicates(subset=['target', 'definition', 'example'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train-dev-test split")
    parser.add_argument('--train_size', type=float, default=0.75)
    parser.add_argument('--test_size', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dbnary_filename', type=str, default='data/dbnary_nl.jsonl')
    parser.add_argument('--output_folder', type=str, default='train-dev-test')
    args = parser.parse_args()

    df = load_data(args, args.dbnary_filename)
    train, dev, test_seen, test_unseen = train_dev_test_split(df, args)

    stem = pathlib.Path(args.dbnary_filename).stem
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    train.to_json(f'{args.output_folder}/train_{stem}.jsonl', orient='records', lines=True)
    dev.to_json(f'{args.output_folder}/dev_{stem}.jsonl', orient='records', lines=True)
    test_seen.to_json(f'{args.output_folder}/test_seen_{stem}.jsonl', orient='records', lines=True)
    test_unseen.to_json(f'{args.output_folder}/test_unseen_{stem}.jsonl', orient='records', lines=True)
