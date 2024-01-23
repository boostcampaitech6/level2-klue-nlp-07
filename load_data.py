import pickle as pickle
import os
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
import ast
from utils.preprocessing import *
from aeda import *

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  

def convert_to_dict(entry):    
  """
  entry['subject_entity'] = ast.literal_eval(entry['subject_entity'])
  entry['object_entity'] = ast.literal_eval(entry['object_entity'])
  return entry
  """
  try:
    entry['subject_entity'] = ast.literal_eval(entry['subject_entity'])
    entry['object_entity'] = ast.literal_eval(entry['object_entity'])
  except (ValueError, SyntaxError):
    # 예외가 발생하면 해당 열의 데이터를 유효한 변환할 수 없다는 것이므로, 해당 열을 삭제하기 위해 None 저장
    entry['subject_entity'] = None
    entry['object_entity'] = None
  return entry


def load_data(dataset_dir, preprocessing_mode, punct_mode, sentence_mode): # mode
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  dataset = Dataset.from_csv(dataset_dir, encoding='UTF-8')#cp949, UTF-8
  #print(len(dataset))
  dataset = dataset.map(convert_to_dict)
  dataset = dataset.filter(lambda x: x['subject_entity'] is not None and x['object_entity'] is not None)    # AEDA 용 그냥 돌려도 됨, None인 경우 열 삭제 
  #print(len(dataset))
  if preprocessing_mode == 'punct_kr' or 'punct_eng' : 
    semantic_sentence = semantic_typing(dataset, punct_mode, sentence_mode)               # semantic query 생성(punct mode만)
  dataset, special_token_list = preprocessing_dataset(dataset, preprocessing_mode)      # mode
  
  return dataset, special_token_list, semantic_sentence 


def tokenized_dataset(dataset, tokenizer, semantic_sentence):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  tokenized_sentences = tokenizer(
      semantic_sentence,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_offsets_mapping=True
      )
  return tokenized_sentences
  
  
