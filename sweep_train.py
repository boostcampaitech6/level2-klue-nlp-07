import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import numpy as np
import random

# 추가 #
import pytz
from datetime import datetime
from transformers import DataCollatorWithPadding    # Dynamic Padding
import wandb        # Wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import EarlyStoppingCallback      # EarlyStopping
import argparse     # argparse
from utils.losses import *        # loss
from utils.metrics import *     # compute_metrics (f1 score, auprc)

import warnings     # Removal Warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support*")
warnings.filterwarnings("ignore", ".*target is close to zero*")


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  
  # train time 설정
  now = datetime.now(pytz.timezone("Asia/Seoul"))
  time = now.strftime('%Y_%m_%d_%H_%M_%S')
  
  set_seed(42)
  
  wandb.init()      # init()을 해야 sweep마다 w&b에서 확인 가능
  config = wandb.config
  MODEL_NAME = config.model_name
  preprocessing_mode = config.preprocessing_mode
  punct_mode = config.preprocessing_mode
  sentence_mode = config.sentence_mode
  
  
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)    
  
  # load dataset
  train_dataset, train_sp_token_list, train_semantic_sentence = load_data("../dataset/train/train.csv", preprocessing_mode, punct_mode, sentence_mode)
  dev_dataset, dev_sp_token_list, dev_semantic_sentence = load_data("../dataset/train/dev.csv", preprocessing_mode, punct_mode, sentence_mode) # validation용 데이터는 따로 만드셔야 합니다.
  sp_token_list = list(set(train_sp_token_list + dev_sp_token_list))
  if sp_token_list is not None :
    print("sp_tokens: ",sp_token_list)
    tokenizer.add_special_tokens({'additional_special_tokens':sp_token_list}) 
  
  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, train_semantic_sentence)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, dev_semantic_sentence)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  print(tokenizer.decode(RE_train_dataset[0]['input_ids']))
  print(tokenizer.decode(RE_dev_dataset[0]['input_ids']))

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  #print(model.config)
  model.parameters
  model.to(device)
  model.resize_token_embeddings(len(tokenizer))
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = "micro f1 score",
    report_to = 'wandb',
    run_name=f'[{time}] {preprocessing_mode}-semantic:{sentence_mode}-{MODEL_NAME}'
  )
  

  trainer = FocalLossTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]        # EarlyStopping
    # fp16 = true
    data_collator=data_collator
  )

  # train model
  trainer.train()
  model.save_pretrained(f'./best_model-sweep-{preprocessing_mode}-semantic:{sentence_mode} [{time}]')
  
def main():
  
  sweep_config = {
      "name": "klue-nlp7-lv2-sweep",
      "method": "bayes",
      "metric": {"goal": "maximize", "name": "eval/micro f1 score"},
      "parameters": {
          "model_name": {"values": ["klue/roberta-large"]}, 
          "preprocessing_mode": {"values": ["punct_kr", "punct_eng"]},
          "num_train_epochs": {"min": 3, "max": 6},
          "learning_rate": {"min": 15e-6, "max": 5e-5},
          "per_device_train_batch_size": {"values":[8, 16, 32]},
          "per_device_eval_batch_size": {"values":[8, 16, 32]},
      }
  }
  
  sweep_id = wandb.sweep(sweep=sweep_config, project="klue-nlp7-lv2")
  wandb.agent(sweep_id, function=train, count=3)
  
  
if __name__ == '__main__':
  main()