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
from datasets import load_dataset, Dataset, IterableDataset, concatenate_datasets
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments


def formatting_func_factory(tokenizer, args):
    system_message = dict()
    system_message['nl'] = "Je bent een lexicograaf die vertrouwd is met het geven van beknopte definities van woordbetekenissen."
    system_message['it'] = "Sei un lessicografo esperto nel fornire definizioni concise dei significati delle parole."
    system_message['sv'] = "Du är en lexikograf som är van vid att ge kortfattade definitioner av ordens betydelser."
    system_message['no'] = "Du er en leksikograf som er kjent med å gi presise definisjoner av ords betydning."
    system_message['es'] = "Eres un lexicógrafo familiarizado con proporcionar definiciones concisas de los significados de las palabras."
    system_message['ja'] = "あなたは、単語の意味の簡潔な定義を提供することに熟練した辞書編纂者です。"
    system_message['de'] = "Du bist ein Lexikograf, der mit der Bereitstellung prägnanter Definitionen von Wortbedeutungen vertraut ist."
    system_message['pt'] = "Você é um lexicógrafo familiarizado com a fornecimento de definições concisas dos significados das palavras."
    system_message['ru'] = "Вы — лексикограф, знакомый с составлением кратких определений значений слов."
    system_message['el'] = "Είστε ένας λεξικογράφος εξοικειωμένος με την παροχή συνοπτικών ορισμών των εννοιών των λέξεων."
    system_message['fr'] = "Vous êtes un lexicographe habitué à fournir des définitions concises des significations des mots."    
    system_message['en'] = "You are a lexicographer familiar with providing concise definitions of word meanings."
    system_message['tr'] = "Sen, kelime anlamlarının özlü tanımlarını sağlamaya aşina bir sözlük yazarsısın."
    system_message['mg'] = "Ianao dia lexicographer mahazatra amin'ny fanomezana fanazavana fohy momba ny dikan'ny teny."
    system_message['da'] = "Du er en leksikograf, der er vant til at give præcise definitioner af ords betydninger."
    system_message['ca'] = "Ets un lexicògraf familiaritzat amb la creació de definicions concises dels significats de les paraules."
    system_message['lt'] = "Jūs esate leksikografas, kuris gerai susipažinęs su trumpų žodžių reikšmių apibrėžimų pateikimu."
    system_message['la'] = "Es lexicographus peritus, qui breves definitiones significatuum verborum praebet."
    system_message['id'] = "Anda adalah seorang leksikograf yang terbiasa memberikan definisi singkat dari makna kata-kata."
    system_message['pl'] = "Jesteś leksykografem, który zna się na podawaniu zwięzłych definicji znaczeń słów."
    system_message['ku'] = "Hûn lexicographer in ku bi dayîna şîroveyên kurt ên maneya peyvên nasnamekî ne."
    system_message['zh'] = "你是一位熟悉提供简明单词含义定义的词典编纂者。"
    system_message['fi'] = "Olet sanakirjantekijä, joka tuntee sanan merkitysten ytimekkäiden määritelmien antamisen."
    
    user_message = dict()
    user_message['nl'] = 'Geef alstublieft een beknopte definitie van de betekenis van het woord "{}" in de volgende zin: {}'
    user_message['it'] = 'Si prega di fornire una definizione concisa per il significato della parola "{}" nella seguente frase: {}'
    user_message['sv'] = 'Vänligen ge en kortfattad definition av betydelsen av ordet "{}" i följande mening: {}'
    user_message['es'] = 'Por favor, proporcione una definición concisa para el significado de la palabra "{}" en la siguiente oración: {}'
    user_message['no'] = 'Vennligst gi en kortfattet definisjon av betydningen av ordet "{}" i den følgende setningen: {}'
    user_message['ja'] = '次の文での「{}」という単語の意味に対する簡潔な定義を提供してください: {}'
    user_message['de'] = 'Bitte geben Sie eine prägnante Definition für die Bedeutung des Wortes "{}" im folgenden Satz an: {}'
    user_message['pt'] = 'Por favor, forneça uma definição concisa para o significado da palavra "{}" na seguinte frase: {}'
    user_message['ru'] = 'Пожалуйста, предоставьте краткое определение значения слова "{}" в следующем предложении: {}'
    user_message['el'] = 'Παρακαλώ παρέχετε έναν συνοπτικό ορισμό για τη σημασία της λέξης "{}" στην παρακάτω πρόταση: {}'
    user_message['fr'] = 'Veuillez fournir une définition concise du sens du mot "{}" dans la phrase suivante : {}'
    user_message['en'] = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'
    user_message['tr'] = 'Lütfen aşağıdaki cümledeki "{}" kelimesinin anlamı için özlü bir tanım sağlayın: {}'
    user_message['mg'] = 'Azafady, omeo fanazavana fohy momba ny dikan\'ny teny "{}" ao amin\'ity fehezanteny manaraka ity: {}'
    user_message['da'] = 'Venligst giv en kortfattet definition af betydningen af ordet "{}" i den følgende sætning: {}'
    user_message['ca'] = 'Si us plau, proporcioneu una definició concisa del significat de la paraula "{}" en la següent frase: {}'
    user_message['lt'] = 'Prašome pateikti trumpą žodžio "{}" reikšmės apibrėžimą šioje sakinyje: {}'
    user_message['la'] = 'Quaeso, praebe brevem definitionem significatuum verbi "{}" in sequenti sententia: {}'
    user_message['id'] = 'Tolong berikan definisi singkat untuk makna kata "{}" dalam kalimat berikut: {}'
    user_message['pl'] = 'Proszę podać zwięzłą definicję znaczenia słowa "{}" w następującym zdaniu: {}'
    user_message['ku'] = 'Ji kerema xwe, daxuyaniya kurt ji bo maneya peyva "{}" di gotarê jêrîn de pêşkêş bikin: {}'
    user_message['zh'] = '请提供单词"{}"在以下句子中的简洁定义：{}'
    user_message['fi'] = 'Ole hyvä ja anna lyhyt määritelmä sanan "{}" merkitykselle seuraavassa lauseessa: {}'

    
    def formatting_func(record):
        language=record['language']
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system_message[language]},
                                              {'role': 'user', 'content': user_message[language].format(record['target'], record['example'])},
                                              {'role': 'assistant', 'content': record['label']}],
                                             tokenize=False)
    return formatting_func


def sample_by_language(dataset, args, max_samples=13000):
    # Identify unique languages present in the dataset.
    languages = set(dataset["language"]) 
    
    sampled_subsets = []
    for lang in languages:
        # Filter the dataset for the current language.
        lang_subset = dataset.filter(lambda example: example["language"] == lang)
        total = len(lang_subset)
        sample_size = min(max_samples, total)
        # Shuffle and select the sample_size number of rows.
        lang_sampled = lang_subset.shuffle(seed=args.seed).select(range(sample_size))
        sampled_subsets.append(lang_sampled)
    
    # Concatenate the sampled subsets into one dataset.
    return concatenate_datasets(sampled_subsets)


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
    train_dataset = load_dataset('json', data_files=args.train_filename, split='train', streaming=args.streaming, trust_remote_code=True).shuffle(seed=args.seed)
    eval_dataset = load_dataset('json', data_files=args.dev_filename, split='train', streaming=args.streaming, trust_remote_code=True).shuffle(seed=args.seed)
    train_dataset = train_dataset.rename_column("definition", "label")
    eval_dataset = eval_dataset.rename_column("definition", "label")


    if " " in args.language and args.verbose:
        print('-- Balancing --')
        train_dataset = sample_by_language(train_dataset, args)
        eval_dataset = sample_by_language(eval_dataset, args)


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
    parser.add_argument('--hugginface_token', type=str, default='YOUR HUGGINFACE TOKEN')
    parser.add_argument('--train_filename', type=str, nargs='+', default='data/train.jsonl')
    parser.add_argument('--dev_filename', type=str, nargs='+', default='data/dev.jsonl')
    parser.add_argument('--qlora', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--cache_dir', type=str, default="cache_dir")
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
