import re
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from trl import SFTTrainer
import evaluate
from huggingface_hub import login
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset, IterableDataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments


def formatting_func_factory(tokenizer, args):
    system_message = dict()
    system_message['nl'] = "Je bent een lexicograaf die vertrouwd is met het geven van beknopte definities van woordbetekenissen."

    user_message = dict()
    user_message['nl'] = 'Geef alstublieft een beknopte definitie van de betekenis van het woord "{}" in de volgende zin: {}'
    
    def formatting_func(record):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system_message[args.language]},
                                              {'role': 'user', 'content': user_message[args.language].format(record['target'], record['example'])},
                                              {'role': 'assistant', 'content': record['label']}],
                                             tokenize=False)
    return formatting_func


def train(args):
    if args.verbose: print('-- Set seed --')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.verbose: print('-- Hugginface login --')
    login(args.hugginface_token)


    if args.verbose: print(f'-- Set accelerator --')
    fsdp_plugin = FullyShardedDataParallelPlugin(  # see: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


    if args.verbose: print('-- Load tokenizer --')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name,
                                              padding_side="right",
                                              add_eos_token=False,
                                              add_bos_token=False,
                                              cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token


    if args.verbose: print('-- Load train dataset --')
    train_dataset = load_dataset('json', data_files=args.train_filename, split='train', streaming=args.streaming, trust_remote_code=True).shuffle(seed=args.seed)#.select(range(100))
    eval_dataset = load_dataset('json', data_files=args.dev_filename, split='train', streaming=args.streaming, trust_remote_code=True).shuffle(seed=args.seed)#.select(range(100))
    train_dataset = train_dataset.rename_column("definition", "label")
    eval_dataset = eval_dataset.rename_column("definition", "label")
        

    if args.verbose: print(f'-- Set tuning parameters [model, device, cache] --')
    settings = dict(pretrained_model_name_or_path=args.base_model_name,
                    device_map='auto',
                    cache_dir=args.cache_dir,
                    trust_remote_code=True)

    
    if args.verbose: print(f'-- QLoRa {"enabled" if args.qlora and args.lora else "disabled"} --')
    if args.qlora and args.lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
            bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # specifies the data type for computation
        )
        settings['quantization_config'] = bnb_config


    if args.verbose: print(f'-- LoRa {"enabled" if args.lora else "disabled"} --')
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM")


    if args.verbose: print(f'-- Load base model --')
    base_model = AutoModelForCausalLM.from_pretrained(**settings)
    base_model.config.use_cache = False  # avoid using cache params
    base_model.gradient_checkpointing_enable()  # this will reduce GPU memory but slow down the process
    base_model = prepare_model_for_kbit_training(base_model)  # see: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    base_model.config.pretraining_tp = 1  # info: https://github.com/huggingface/transformers/pull/24906
    if not args.lora: model = base_model
    else: model = get_peft_model(base_model, peft_config)
    model = accelerator.prepare_model(model, device_placement=True)


    if args.verbose: print(f'-- Set early stopping --')
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
        
    rouge = evaluate.load("rouge")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the preds as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {"rougeL": result["rougeL"]}
        
    callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold)]
    
    
    if args.verbose: print(f'-- Set SFTTrainer --')
    output_dir = Path(f'{args.output_dir}/{args.finetuned_model_name}{args.tag}')
    logging_dir = str(output_dir.parent) + f'/log_{output_dir.name}'
    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        packing=True,
        formatting_func=formatting_func_factory(tokenizer, args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # useful for compute_metrics
        args=TrainingArguments(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False}, # avoid warning
            output_dir=output_dir,
            logging_dir=logging_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            do_eval=True,
            bf16=True,
            metric_for_best_model="rougeL", 
            greater_is_better=True, 
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim="paged_adamw_8bit",
            evaluation_strategy='epoch',
            gradient_accumulation_steps=args.gradient_accumulation_steps, # see: https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260/5
            max_steps=-1 if not args.streaming else (args.num_rows // args.batch_size) // args.gradient_accumulation_steps * args.num_train_epochs,
            save_strategy='epoch',
            save_total_limit=1, # keep only the best model
            load_best_model_at_end=True,  # load the best model at the end
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )


    if args.verbose: print(f'-- Training is started! --')
    trainer.train()

    if args.verbose: print(f'-- Store final model --')
    trainer.model.save_pretrained(str(output_dir) + f'/checkpoint-{trainer.state.best_model_checkpoint.split("-")[-1]}', save_embedding_layers=True)
    trainer.tokenizer.save_pretrained(str(output_dir) + f'/checkpoint-{trainer.state.best_model_checkpoint.split("-")[-1]}')
    pd.DataFrame(trainer.state.log_history).to_csv(str(output_dir) + f'/log.tsv', sep='\t', index=False)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Finetuning')
    parser.add_argument('--language', type=str, default='nl')
    parser.add_argument('--streaming', action='store_true', help='Load a large dataset as IterableDataset')
    parser.add_argument('--base_model_name', type=str)
    parser.add_argument('--hugginface_token', type=str, default='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')
    parser.add_argument('--train_filename', type=str, nargs='+', default='data/train.jsonl')
    parser.add_argument('--dev_filename', type=str, nargs='+', default='data/dev.jsonl')
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--cache_dir', type=str, default="/mimer/NOBACKUP/groups/cik_data/fra_hf_cache")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetuned_model_name', type=str)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_rows', type=int, default=-1)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001)
    args = parser.parse_args()
  
    train(args)
