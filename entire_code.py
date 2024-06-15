# 런타임 GPU인지 확인하기!

!pip install -U evaluate accelerate sacrebleu

from huggingface_hub import notebook_login
notebook_login()

import os

import re
import pandas as pd
import numpy as np

from datasets import load_dataset, load_metric, DatasetDict, Dataset

import transformers
from transformers import TextDataset, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments

import evaluate

import random
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

checkpoint = "DezS/Accent-and-accentless-Vietnamese-dataset"
raw_dataset = load_dataset(checkpoint)

# It takes too long time for huge dataset, we need sampling.
raw_dataset['train'] = raw_dataset['train'].shuffle(seed=42).select(range(10000))

# removing accents with regex
def remove_accents(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s
  
# remove all accents remaining in dataset
updated_dataset = raw_dataset.map(lambda example: {"AccentRemovedSentences": remove_accents(example["AccentlessSentences"])},
                                  remove_columns=["AccentlessSentences"],
                                  num_proc=32,
                                  )

# train-val-test split
updated_dataset = updated_dataset['train'].train_test_split(test_size=0.25, seed=42)
valid_test_dataset = updated_dataset['test'].train_test_split(test_size=0.5, seed=42)
splitted_dataset = DatasetDict({
    'train': updated_dataset['train'],
    'val': valid_test_dataset['train'],
    'test': valid_test_dataset['test'],
    })
splitted_dataset

tokenizer = GPT2Tokenizer.from_pretrained('NlpHUST/gpt2-vietnamese')
model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese')
model = model.to("cuda:0")

# padding token
tokenizer.pad_token = tokenizer.eos_token

# tokenizing the whole dataset (we use same tokenizer for two columns!)
def tokenize_function(example):
    return tokenizer([example["AccentRemovedSentences"], example["Sentences"]], truncation=True, 
                     padding='max_length', max_length=64, return_special_tokens_mask=True)

transformers.logging.set_verbosity_error()
splitted_dataset['val'] = splitted_dataset['val'].map(tokenize_function, 
                                                      remove_columns=["Sentences", "AccentRemovedSentences"],
                                                      num_proc=32) # multiprocessing
splitted_dataset['test'] = splitted_dataset['test'].map(tokenize_function, 
                                                        remove_columns=["Sentences", "AccentRemovedSentences"],
                                                        num_proc=32)
splitted_dataset['train'] = splitted_dataset['train'].map(tokenize_function, 
                                                          remove_columns=["Sentences", "AccentRemovedSentences"],
                                                          num_proc=32)

# # 사전학습된 모델은 기존의 프롬프트의 뒤에 이어질 내용을 생성하는 모델이다.
# example_prompt = "hôm qua, chúng ta" # "어제, 우리는"
# input_ids = tokenizer.encode(example_prompt, return_tensors='pt').to("cuda:0")
# output_ids = model.generate(input_ids).to("cuda:0")
# tokenizer.decode(*output_ids)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                               mlm=False) # GPT-2 does not use masked language modeling

os.getcwd()
output_dir = os.path.join(os.getcwd(), "Vietnamese_Accent_AI")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    # overwrite_output_dir=False, # Default=False
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=1e-4,
    save_steps=10_000,
    save_strategy='steps',
    evaluation_strategy='steps',
    report_to="wandb",  # enable logging to W&B
    run_name="vietnamese-accent-ai-10000",  # name of the W&B run (optional)
    logging_steps=1,  # how often to log to W&B
    save_total_limit=3,
    push_to_hub=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
)

def compute_metrics(pred):
    pass # to be determined

trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=splitted_dataset['train'],
        eval_dataset=splitted_dataset['val'],
        # compute_metrics= #to be determined
    )

trainer.train()
trainer.save_model(output_dir=output_dir)

# inference of new fine-tuned model
test_prompt = "Ban da hoc bai tap xong chua?" # "너 숙제는 다 끝냈니?"

input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to("cuda:0")
output_ids = model.generate(input_ids).to("cuda:0")
tokenizer.decode(*output_ids)

# inference of new fine-tuned model
trainer.predict(test_dataset=splitted_dataset['test'])
