import pickle as pickle
import os
import pandas as pd
import torch

# Custom imports & variables
import re
import datetime
ts = datetime.datetime.now().strftime("%m%d%H%M%S")


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

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  subject_type = []
  object_type = []
  sentences = []
  for i,j,k in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):

    # 1. Entity에서 type만 추출해서 컬럼 추가
    s_type = i.split('\': ')[4][1:-2]
    o_type = j.split('\': ')[4][1:-2]
    subject_type.append(s_type)
    object_type.append(o_type)

    # 2. Entity에서 word만 추출
    i = i.split('start_idx')[0].split('word')[1][4:-4] # 수정된 부분
    j = j.split('start_idx')[0].split('word')[1][4:-4] # 수정된 부분
    subject_entity.append(i)
    object_entity.append(j)
    
    # 3-1. Sentence 스페셜 토큰 추가된 형태로수정
    # k = re.sub(i, f"[S:{s_type}]{i}[/S:{s_type}]", k)
    # k = re.sub(j, f"[O:{o_type}]{j}[/O:{o_type}]", k)

    # 3-2. Symbol token으로 Entity 및 Type 표시
    k = re.sub(i, f"@*{s_type}*{i}", k)
    k = re.sub(j, f"#^{o_type}^{j}", k)

    sentences.append(k)

    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,'subject_entity':subject_entity,'object_entity':object_entity,'subject_type':subject_type,'object_type':object_type,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02, t01, t02 in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subject_type'], dataset['object_type']):
    temp = ''
    temp = f"@{e01}와 #{e02}의 관계는 *{t01}*과 ^{t02}^의 관계이다."
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  tokenized_test = tokenizer.decode(tokenized_sentences['input_ids'][0])
  print(tokenized_test)
  with open(f"./logs/tokenized_test_{ts}.txt", "w") as text_file:
    text_file.write(tokenized_test)
  return tokenized_sentences
