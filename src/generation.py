import time
import torch
import pathlib
import argparse
from peft import PeftModel
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

def load_model(args, tokenizer):
    login(args.hugginface_token)
    settings = dict(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    device_map='auto')

    if args.quantization:
        settings['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(**settings)
    model.eval()

    if args.peft_model_name_or_path!= "None":
        peft_model = PeftModel.from_pretrained(model, args.peft_model_name_or_path)
        peft_model.eval()
        return peft_model.merge_and_unload()
    else:
        return model

def load_tokenizer(args):
    login(args.hugginface_token)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              padding_side="left",
                                              #use_fast=False,
                                              add_eos_token=True,
                                              add_bos_token=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
    system_message['en'] = "You are a lexicographer familiar with providing concise definitions of word meanings."
    system_message['tr'] = "Sen, kelime anlamlarının özlü tanımlarını sağlamaya aşina bir sözlük yazarsısın."
    system_message['mg'] = "Ianao dia lexicographer mahazatra amin'ny fanomezana fanazavana fohy momba ny dikan'ny teny."
    system_message['da'] = "Du er en leksikograf, der er vant til at give præcise definitioner af ords betydninger."
    system_message['ca'] = "Ets un lexicògraf familiaritzat amb la creació de definicions concises dels significats de les paraules."
    system_message['fr'] = "Vous êtes un lexicographe habitué à fournir des définitions concises des significations des mots."
    system_message['lt'] = "Jūs esate leksikografas, kuris gerai susipažinęs su trumpų žodžių reikšmių apibrėžimų pateikimu."
    system_message['la'] = "Es lexicographus peritus, qui breves definitiones significatuum verborum praebet."
    system_message['id'] = "Anda adalah seorang leksikograf yang terbiasa memberikan definisi singkat dari makna kata-kata."
    system_message['pl'] = "Jesteś leksykografem, który zna się na podawaniu zwięzłych definicji znaczeń słów."
    system_message['ku'] = "Hûn lexicographer in ku bi dayîna şîroveyên kurt ên maneya peyvên nasnamekî ne."
    system_message['el'] = "Είστε ένας λεξικογράφος εξοικειωμένος με την παροχή συνοπτικών ορισμών των εννοιών των λέξεων."
    system_message['zh'] = "你是一位熟悉提供简明单词含义定义的词典编纂者。"
    system_message['fi'] = "Olet sanakirjantekijä, joka tuntee sanan merkitysten ytimekkäiden määritelmien antamisen."
    system_message['ru'] = "Вы — лексикограф, знакомый с составлением кратких определений значений слов."

    user_message = dict()
    user_message['nl'] = 'Geef alstublieft een beknopte definitie van de betekenis van het woord "{}" in de volgende zin: {}'
    user_message['it'] = 'Si prega di fornire una definizione concisa per il significato della parola "{}" nella seguente frase: {}'
    user_message['sv'] = 'Vänligen ge en kortfattad definition av betydelsen av ordet "{}" i följande mening: {}'
    user_message['es'] = 'Por favor, proporcione una definición concisa para el significado de la palabra "{}" en la siguiente oración: {}'
    user_message['no'] = 'Vennligst gi en kortfattet definisjon av betydningen av ordet "{}" i den følgende setningen: {}'
    user_message['ja'] = '次の文での「{}」という単語の意味に対する簡潔な定義を提供してください: {}'
    user_message['de'] = 'Bitte geben Sie eine prägnante Definition für die Bedeutung des Wortes "{}" im folgenden Satz an: {}'
    user_message['pt'] = 'Por favor, forneça uma definição concisa para o significado da palavra "{}" na seguinte frase: {}'
    user_message['en'] = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'
    user_message['tr'] = 'Lütfen aşağıdaki cümledeki "{}" kelimesinin anlamı için özlü bir tanım sağlayın: {}'
    user_message['mg'] = 'Azafady, omeo fanazavana fohy momba ny dikan\'ny teny "{}" ao amin\'ity fehezanteny manaraka ity: {}'
    user_message['da'] = 'Venligst giv en kortfattet definition af betydningen af ordet "{}" i den følgende sætning: {}'
    user_message['ca'] = 'Si us plau, proporcioneu una definició concisa del significat de la paraula "{}" en la següent frase: {}'
    user_message['fr'] = 'Veuillez fournir une définition concise du sens du mot "{}" dans la phrase suivante : {}'
    user_message['lt'] = 'Prašome pateikti trumpą žodžio "{}" reikšmės apibrėžimą šioje sakinyje: {}'
    user_message['la'] = 'Quaeso, praebe brevem definitionem significatuum verbi "{}" in sequenti sententia: {}'
    user_message['id'] = 'Tolong berikan definisi singkat untuk makna kata "{}" dalam kalimat berikut: {}'
    user_message['pl'] = 'Proszę podać zwięzłą definicję znaczenia słowa "{}" w następującym zdaniu: {}'
    user_message['ku'] = 'Ji kerema xwe, daxuyaniya kurt ji bo maneya peyva "{}" di gotarê jêrîn de pêşkêş bikin: {}'
    user_message['el'] = 'Παρακαλώ παρέχετε έναν συνοπτικό ορισμό για τη σημασία της λέξης "{}" στην παρακάτω πρόταση: {}'
    user_message['zh'] = '请提供单词"{}"在以下句子中的简洁定义：{}'
    user_message['fi'] = 'Ole hyvä ja anna lyhyt määritelmä sanan "{}" merkitykselle seuraavassa lauseessa: {}'
    user_message['ru'] = 'Пожалуйста, предоставьте краткое определение значения слова "{}" в следующем предложении: {}'

    def formatting_func(record):
        language = record['language'] # args.language
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system_message[language]},
                                              {'role': 'user', 'content': user_message[language].format(record['target'], record['example'])}],
                                               tokenize=False, add_generation_prompt=True)
    return formatting_func

def generation(pipe, dataset, args):
    formatting_func = formatting_func_factory(pipe.tokenizer, args)
    prompts = [formatting_func(row) for row in dataset]

    tokens = ['.', ' .']
    eos_tokens = [pipe.tokenizer.eos_token_id] + [pipe.tokenizer.encode(token, add_special_tokens=False)[0]
                                                  for token in tokens]
    
    outputs = list()
    start_time = time.time()
    for out in pipe(prompts,
                    forced_eos_token_id = eos_tokens,
                    max_time = args.max_time * args.batch_size,
                    repetition_penalty = args.repetition_penalty,
                    eos_token_id = eos_tokens,
                    max_new_tokens = args.max_new_tokens,
                    truncation = True,
                    batch_size = args.batch_size,
                    pad_token_id = pipe.tokenizer.eos_token_id):
        outputs.append(format_output(out, args))
    print('Generation time:', time.time() - start_time)
    print('# inputs:', len(outputs))
    return outputs

def format_output(text, args):
    #if args.pretrained_model_name_or_path == 'Unbabel/TowerInstruct-7B-v0.2'
    text = " ".join(text[0]["generated_text"].split()).strip() + '\n' # todo .
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generation')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='Unbabel/TowerInstruct-7B-v0.2')
    parser.add_argument('--peft_model_name_or_path', type=str, default="None")
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--hugginface_token', type=str, default='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')
    parser.add_argument('--test_filename', type=str, nargs='+', default=['data/test.jsonl'])
    parser.add_argument('--max_time', type=float, default=4.5)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--length_penalty', type=float, default=1.1)
    parser.add_argument('--output_dir', type=str, default='models')
    args = parser.parse_args()
    
    print('CUDA:', torch.cuda.is_available())
    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    for filename in args.test_filename:
        stem = pathlib.Path(filename).stem
        if pathlib.Path(f'{args.output_dir}/{stem}.txt').exists():
            print('File exists:', f'{args.output_dir}/{stem}.txt')
            continue
        
        datasets = load_dataset('json', data_files=filename, split='train')
        outputs = generation(pipe, datasets, args)

        
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{args.output_dir}/{stem}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(outputs)
