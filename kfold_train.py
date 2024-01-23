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

from sklearn.model_selection import StratifiedKFold

import warnings     # Removal Warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support*")
warnings.filterwarnings("ignore", ".*target is close to zero*")

def set_seed(seed: int = 42):
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


def train(args, train_dataset, dev_dataset, fold):

    set_seed(42)
    
    MODEL_NAME = args.model_name
    preprocessing_mode = args.preprocessing_mode
    sentence_mode = args.sentence_mode
    train_dataset.to_csv("./dataset.csv")
    dev_dataset.to_csv("./devset.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)    
    
    # load dataset
    train_dataset, train_sp_token_list, train_semantic_sentence  = load_data("./dataset.csv", preprocessing_mode, preprocessing_mode, sentence_mode)
    dev_dataset, dev_sp_token_list, dev_semantic_sentence  = load_data("./devset.csv", preprocessing_mode, preprocessing_mode, sentence_mode) # validation용 데이터는 따로 만드셔야 합니다.
    sp_token_list = list(set(train_sp_token_list + dev_sp_token_list))
    if sp_token_list is not None :
        tokenizer.add_special_tokens({'additional_special_tokens':sp_token_list}) 
    
    
    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, train_semantic_sentence )
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, dev_semantic_sentence )

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    
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
        learning_rate=1.5e-5,               # learning_rate
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
        eval_steps = 1000,            # evaluation step.1082
        load_best_model_at_end = True,
        metric_for_best_model = "micro f1 score",
        report_to = "wandb",
        run_name=f'[{time}] {preprocessing_mode}-semantic-n_fold:{fold}'
    )


    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(f'./best_model-sweep-fold{fold}-{time}')
    wandb.finish()  # wandb 세션 종료

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="klue/roberta-large")
    parser.add_argument('--preprocessing_mode', type=str, default="punct_eng")
    parser.add_argument('--sentence_mode', type=str, default="1")
    args = parser.parse_args()
    
        
    dataset = pd.read_csv("../dataset/train/train.csv")
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, dev_idx) in enumerate(kfold.split(dataset, dataset['label'])):
        wandb.init(project= 'klue-nlp-lv2', name = f'semantic-n_split:{fold}')
        print(f"{fold + 1} 번째 폴드 진행 중")

        train_fold = dataset.iloc[train_idx]
        dev_fold = dataset.iloc[dev_idx]

        train(args, train_fold, dev_fold, fold)


if __name__ == '__main__':
    main()