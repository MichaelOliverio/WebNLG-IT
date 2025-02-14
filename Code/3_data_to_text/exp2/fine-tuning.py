#!pip install -q accelerate
#!pip install peft
#!pip install bitsandbytes
#!pip install transformers
#!pip install trl
#!pip install huggingface_hub
#!pip install accelerate
#!pip install --upgrade torch

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import pandas as pd
from datasets import Dataset
import re
import random
import gc
from huggingface_hub import login

# hugging face
login(token="...")

# Paths for datasets and models
data_path_datasets = {
    'en': '.\\datasets\\en',
    'it': '.\\datasets\\it',
    'ge': '.\\datasets\\ge',
    'ru': '.\\datasets\\ru',
}

data_path_models = {
    'en': '.\\models\\en',
    'it': '.\\models\\it',
    'ge': '.\\models\\ge',
    'ru': '.\\models\\ru',
}

# List of languages and model names
langs = [
    'en',
    'it',
    'ge',
    'ru',
]
models_name = {
    'en': [
        'mistralai/Mistral-Nemo-Instruct-2407', # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        'meta-llama/Llama-3.1-8B-Instruct', # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        'Qwen/Qwen2.5-7B-Instruct', #https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        #'ibm-granite/granite-3.1-8b-instruct', # https://huggingface.co/ibm-granite/granite-3.1-8b-instruct
    ],
    'it': [
        'mistralai/Mistral-Nemo-Instruct-2407', # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        'meta-llama/Llama-3.1-8B-Instruct', # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        'Qwen/Qwen2.5-7B-Instruct', #https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        #'ibm-granite/granite-3.1-8b-instruct', # https://huggingface.co/ibm-granite/granite-3.1-8b-instruct
    ],
    'ge': [
        'mistralai/Mistral-Nemo-Instruct-2407', # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        'meta-llama/Llama-3.1-8B-Instruct', # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        'Qwen/Qwen2.5-7B-Instruct', #https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        #'ibm-granite/granite-3.1-8b-instruct', # https://huggingface.co/ibm-granite/granite-3.1-8b-instruct
    ],
    'ru': [
        'mistralai/Mistral-Nemo-Instruct-2407', # https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
        'meta-llama/Llama-3.1-8B-Instruct', # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        'Qwen/Qwen2.5-7B-Instruct', #https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
        #'ibm-granite/granite-3.1-8b-instruct', # https://huggingface.co/ibm-granite/granite-3.1-8b-instruct
    ],
}

################################################################################
# QLoRA parameters
################################################################################
lora_r = 64 # LoRA attention dimension
lora_alpha = 16 # Alpha parameter for LoRA scaling
lora_dropout = 0.1 # Dropout probability for LoRA layers

################################################################################
# bitsandbytes parameters
################################################################################
use_4bit = True # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)

################################################################################
# TrainingArguments parameters
################################################################################
num_train_epochs = 2 # Number of training epochs
output_dir = "./results" # Output directory where the model predictions and checkpoints will be stored
fp16 = False # Enable fp16 training
bf16 = True # Enable bf16 training (set bf16 to True with an A100)
per_device_train_batch_size = 4 # Batch size per GPU for training
per_device_eval_batch_size = 4 # Batch size per GPU for evaluation
gradient_accumulation_steps = 1 # Number of update steps to accumulate the gradients for
gradient_checkpointing = True # Enable gradient checkpointing
max_grad_norm = 0.3 # Maximum gradient normal (gradient clipping)
learning_rate = 2e-4 # Initial learning rate (AdamW optimizer)
weight_decay = 0.001 # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit" # Optimizer to use
lr_scheduler_type = "cosine" # Learning rate schedule
max_steps = -1 # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03 # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = True # Group sequences into batches with same length. Saves memory and speeds up training considerably
save_steps = 0 # Save checkpoint every X updates steps
logging_steps = 25 # Log every X updates steps

################################################################################
# SFT parameters
################################################################################
max_seq_length = None # Maximum sequence length to use
packing = False # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0} # Load the entire model on the GPU 0

def prepare_dataset(lang, split):
    """Prepares dataset for the specified language and split."""
    df = pd.read_csv(f'{data_path_datasets[lang]}\\{split}.csv')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    dataset = []
    for index, row in df.iterrows():
        dataset.append({
            #'instruction': f'Given the following triples, generate the corresponding text. Triples=[{row["data_unit"]}]',
            #'instruction': f'<s> [INST] Given the following triples in (TRIPLE), you have to generate the corresponding text in (ANW) [/INST] [TRIPLE] {row["data_unit"]} [/TRIPLE]', #[ANW] {row['sentence']} [/ANW] </s>
            'instruction': f'Given the following triples in (TRIPLE), you have to generate the corresponding text in (ANW)',
            'eid': row['eid'],
            'input': row['data_unit'],
            'output': row['sentence'],
        })

    return Dataset.from_pandas(pd.DataFrame(dataset, columns=['instruction', 'eid', 'input', 'output']))


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

def load_model(model_name):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    eval_strategy="steps",  # Calcolo della validation loss ad ogni step
    save_strategy="steps",  # Checkpoint del modello ad ogni step
    save_total_limit=3,  # Limita il numero di checkpoint salvati
    load_best_model_at_end=True,  # Carica il modello migliore al termine dell'addestramento
    metric_for_best_model="eval_loss",  # Sceglie la metrica per determinare il miglior modello
    greater_is_better=False,  # Indica se un valore più alto della metrica è migliore o no
    eval_steps=500,  # Numero di passaggi prima di valutare il modello
    logging_steps= 500,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        #text = f"### Question: {example['instruction'][i]}\n ### Answer: <s>{example['output'][i]}</s>"
        #text = f"{example['instruction'][i]} [ANW] {example['output'][i]} [/ANW] </s>"
        text = f"<s> [INST] {example['instruction'][i]} [/INST] [TRIPLE] {example['input'][i]} [/TRIPLE] [ANW] {example['output'][i]} [/ANW] </s>"

        output_texts.append(text)

    return output_texts

def train(model, tokenizer, train_dataset, eval_dataset):
    #response_template = " ### Answer:"
    response_template = " [ANW]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # Train model
    trainer.train()

    return trainer

def load_fine_tuned_model(model_name, fine_tuned_model_path):
  # Reload model in FP16 and merge it with LoRA weights
  base_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      low_cpu_mem_usage=True,
      return_dict=True,
      torch_dtype=torch.float16,
      device_map=device_map,
  )

  model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
  model = model.merge_and_unload()

  # Reload tokenizer to save it
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  return model, tokenizer

def clear_output(output):
    #match = re.search(r'<s>(.*?)</s>', output)
    match = re.search(r'\[ANW\](.*?)\[/ANW\]', output, re.DOTALL)
    return match.group(1) if match else "Output not found"

def model_generation(pipe, test_dataset, output_path):
    eids = []
    inputs = []
    predictions = []
    actuals = []
    generations = []

    i = 1
    for record in test_dataset:
      ### Question: {example['instruction'][i]}\n ### Answer:
      print(f'record #{i}')

      if not pipe:
        #max_length = int(len(f" ### Question: {record['instruction']}\n ### Answer:<s>{record['output']}</s>") * 1.5)
        #max_length = int(len(f"{record['instruction']} [ANW] {record['output']} [/ANW] </s>") * 1.5)
        max_length = int(len(f"<s> [INST] {record['instruction']} [/INST] [TRIPLE] {record['input']} [/TRIPLE] [ANW] {record['output']} [/ANW] </s>") * 1.5)

        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length, temperature=0.1)

      #result = pipe(f" ### Question: {record['instruction']}\n ### Answer:")
      result = pipe(f"<s> [INST] {record['instruction']} [/INST] [TRIPLE] {record['input']} [/TRIPLE] [ANW]")
      prediction = clear_output(result[0]['generated_text'])

      print(f'eid: {record["eid"]}')
      print(f'input: {record["input"]}')
      print(f'prediction: {prediction}')
      print(f'actual: {record["output"]}')
      print(f'generation: {result[0]["generated_text"]}')

      eids.append(record['eid'])
      inputs.append(record['input'])
      actuals.append(record['output'])
      predictions.append(prediction)
      generations.append(result[0]['generated_text'])

      df = pd.DataFrame(list(zip(eids, inputs, predictions, actuals, generations)), columns=['eids', 'input', 'prediction', 'actual', 'generation'])
      df.to_csv(output_path, index=False)

      print('\n')
      i = i + 1

for lang in langs:
    # Prepare datasets
    train_dataset = prepare_dataset(lang, 'train')
    eval_dataset = prepare_dataset(lang, 'dev')
    test_dataset = prepare_dataset(lang, 'test')

    for model_name in models_name[lang]:
        clean_model_name = model_name.split('/')[1]
        print(f'Language: {lang}, Model: {clean_model_name}')
        output_path = f'{data_path_models[lang]}\\fine-tuned-{clean_model_name}-{lang}-exp2'

        # Load model
        model, tokenizer = load_model(model_name)

        # Train model
        trainer = train(model, tokenizer, train_dataset, eval_dataset)

        # Save fine-tuned model
        trainer.save_model(f'{output_path}')

        # Load fine-tuned model
        fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_name, f'{output_path}')

        # Evaluate fine-tuned model
        pipe = pipeline('text-generation', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=500, temperature=0.1, device=0)
        model_generation(pipe, test_dataset, f'{output_path}-decoding.csv')

        # Clear memory
        del model
        del tokenizer
        del fine_tuned_model
        del fine_tuned_tokenizer
        del pipe
        del trainer
        gc.collect()
        torch.cuda.empty_cache()