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

# Custom imports
from config.config import config
import datetime
from utils.special_tokens import special_tokens

# Custom args
ts = datetime.datetime.now().strftime("%m%d%H%M%S")
os.environ["WANDB_PROJECT"] = "LV2-RE"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
exp_theme = "weighted-ce-loss"
wandb_name = f"{config.model_name[:14]}-{exp_theme}-lr{config.learning_rate}-{config.max_epoch}e"

# Custom loss function class
class FocalLossTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    # focal loss
    # 30 labels
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    
    alpha = 0.25
    gamma = 2.0
    
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))

    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss

    return (focal_loss, outputs) if return_outputs else focal_loss

def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('./utils/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  set_seed(42)
  # load model and tokenizer
  MODEL_NAME = config.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data(config.train_path)
  dev_dataset = load_data(config.dev_path) # validation용 데이터는 따로 만드셔야 합니다.

  # Add special token
  added_token_num = 0
  added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.resize_token_embeddings(tokenizer.vocab_size + added_token_num) # 추가한 special token만큼 model의 embedding 사이즈 변경
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=f'./results/{ts}',          # output directory
    save_total_limit=3,              # number of total save model.
    save_steps=1000,                 # model saving step.
    num_train_epochs=config.max_epoch,              # total number of training epochs
    learning_rate=config.learning_rate,               # learning_rate
    per_device_train_batch_size=config.batch_size,  # batch size per device during training
    per_device_eval_batch_size=config.batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=config.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    save_strategy='steps',
    eval_steps = 1000,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model = 'micro f1 score',
    report_to="wandb",
    run_name=wandb_name
  )

  trainer = FocalLossTrainer( # Custom Loss Trainer 로 instance 정의
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
  
def main():
  train()

if __name__ == '__main__':
  main()
