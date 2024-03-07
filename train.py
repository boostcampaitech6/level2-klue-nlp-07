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
  with open('./utils/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label


def train(args):
  set_seed(42)
  # wandb.init()
  MODEL_NAME = args.model_name
  preprocessing_mode = args.preprocessing_mode
  sentence_mode = args.sentence_mode
  
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)    
  
  # load dataset
  train_dataset, train_sp_token_list, train_semantic_sentence  = load_data("../dataset/train/train.csv", preprocessing_mode, preprocessing_mode, sentence_mode)
  dev_dataset, dev_sp_token_list, dev_semantic_sentence  = load_data("../dataset/train/dev.csv", preprocessing_mode, preprocessing_mode, sentence_mode) # validation용 데이터는 따로 만드셔야 합니다.
  
  # special token 추가
  sp_token_list = list(set(train_sp_token_list + dev_sp_token_list))
  if sp_token_list is not None :
    print("sp_tokens: ",sp_token_list)
    tokenizer.add_special_tokens({'additional_special_tokens':sp_token_list}) 
  
  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, train_semantic_sentence )
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, dev_semantic_sentence )

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  #print(model.config)
  model.parameters
  model.to(device)
  model.resize_token_embeddings(len(tokenizer))
  
  # train time 설정
  now = datetime.now(pytz.timezone("Asia/Seoul"))
  time = now.strftime('%Y_%m_%d_%H_%M_%S')
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=1000,                 # model saving step.
    num_train_epochs=4,              # total number of training epochs
    learning_rate=18e-6,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 1000,
    # save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = "micro f1 score",
    report_to = 'wandb',
    run_name=f'[{time}] {preprocessing_mode}-semantic-{MODEL_NAME}'
  )

  trainer = FocalLossTrainer(  #FocalLossTrainer, LabelSmoothingLossTrainer(classes=30, smoothing=0.1)
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],        # EarlyStopping
    # fp16 = true
    data_collator=data_collator
  )

  # train model
  trainer.train()
  model.save_pretrained(f'./best_model-{MODEL_NAME}-{preprocessing_mode} [{time}]')
  
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="klue/roberta-large") # klue/bert-base, snunlp/KR-SBERT-V40K-klueNLI-augSTS, klue/roberta-large
  parser.add_argument('--preprocessing_mode', type=str, default="punct_kr")# punct_kr, punct_eng (semantic 가능) / default, entity_masking(불가능)
  parser.add_argument('--sentence_mode', type=str, default="1")# "1","2"
  args = parser.parse_args()
  train(args)
  
  
if __name__ == '__main__':
  main()