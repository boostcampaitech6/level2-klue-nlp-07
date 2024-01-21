import pandas as pd

# style: to punctuation english
def processing_special_punct_eng_tokens(dataset) :
  sentences = []
  subject_entity = []
  object_entity = []

  for data in dataset:
        sentence = data['sentence']

        object_start =  int(data['object_entity']['start_idx'])
        object_end =  int(data['object_entity']['end_idx'])
        subject_start =  int(data['subject_entity']['start_idx'])
        subject_end =  int(data['subject_entity']['end_idx'])
        otype = data['object_entity']['type']
        stype = data['subject_entity']['type']
        subject_entity_str = data['subject_entity']['word']
        object_entity_str = data['object_entity']['word']

        if object_start < subject_start:
            new_sentence = sentence[:object_start] + ' # ^ ' + str(otype) + ' ^ '+ sentence[object_start:object_end+1] + ' # ' + sentence[object_end+1:subject_start] + ' @ * '+str(stype)+' * ' + sentence[subject_start:subject_end+1] + ' @ ' + sentence[subject_end+1:]
        else:
            new_sentence = sentence[:subject_start] + ' @ * ' +str(stype)+' * '+ sentence[subject_start:subject_end+1] + ' @ ' + sentence[subject_end+1:object_start] + ' # ^ ' +str(otype)+' ^ '+ sentence[object_start:object_end+1] + ' # ' + sentence[object_end+1:]

        new_subject_ent = subject_entity_str
        new_object_ent = object_entity_str
        
        # 본문 저장
        sentences.append(new_sentence)

        # entity 저장
        subject_entity.append(new_subject_ent)
        object_entity.append(new_object_ent)
  
  return sentences, subject_entity, object_entity
  
# style: to punctuation korean
def processing_special_punct_kr_tokens(dataset) :
  sentences = []
  subject_entity = []
  object_entity = []
  type_mapping = {'PER': '사람', 'ORG': '기관', 'DAT': '날짜', 'LOC': '위치', 'POH': '기타', 'NOH': '수량'}

  for data in dataset:
        sentence = data['sentence']

        object_start =  int(data['object_entity']['start_idx'])
        object_end =  int(data['object_entity']['end_idx'])
        subject_start =  int(data['subject_entity']['start_idx'])
        subject_end =  int(data['subject_entity']['end_idx'])
        otype = data['object_entity']['type']
        stype = data['subject_entity']['type']
        subject_entity_str = data['subject_entity']['word']
        object_entity_str = data['object_entity']['word']

        if object_start < subject_start:
            new_sentence = sentence[:object_start] + ' # ^ ' + type_mapping[otype] + ' ^ '+ sentence[object_start:object_end+1] + ' # ' + sentence[object_end+1:subject_start] + ' @ * '+type_mapping[stype]+' * ' + sentence[subject_start:subject_end+1] + ' @ ' + sentence[subject_end+1:]
        else:
            new_sentence = sentence[:subject_start] + ' @ * ' +type_mapping[stype]+' * '+ sentence[subject_start:subject_end+1] + ' @ ' + sentence[subject_end+1:object_start] + ' # ^ ' +type_mapping[otype]+' ^ '+ sentence[object_start:object_end+1] + ' # ' + sentence[object_end+1:]

        new_subject_ent = subject_entity_str
        new_object_ent = object_entity_str
        
        # 본문 저장
        sentences.append(new_sentence)

        # entity 저장
        subject_entity.append(new_subject_ent)
        object_entity.append(new_object_ent)
  
  return sentences, subject_entity, object_entity

# style: masking entity
def processing_special_entity_masking_tokens(dataset) :
  sentences = []
  subject_entity = []
  object_entity = []
  special_token_list = []
  for data in dataset:
        sentence = data['sentence']

        object_start =  int(data['object_entity']['start_idx'])
        object_end =  int(data['object_entity']['end_idx'])
        subject_start =  int(data['subject_entity']['start_idx'])
        subject_end =  int(data['subject_entity']['end_idx'])
        otype = data['object_entity']['type']
        stype = data['subject_entity']['type']
        
        

        if object_start < subject_start:
            new_sentence = sentence[:object_start] + '<O-' + str(otype) + '>'  + sentence[object_end+1:subject_start] + '<S-'+str(stype)+'>'  + sentence[subject_end+1:]
        else:
            new_sentence = sentence[:subject_start] + '<S-'+str(stype)+'>' + sentence[subject_end+1:object_start] + '<O-'+str(otype)+'>' + sentence[object_end+1:]

        new_subject_ent = '<S-' + str(stype)+ '>'
        new_object_ent = '<O-' + str(otype) + '>'
        if new_subject_ent not in special_token_list : special_token_list.append(new_subject_ent)
        if new_object_ent not in special_token_list : special_token_list.append(new_object_ent)

        # 본문 저장
        sentences.append(new_sentence)

        # entity 저장
        subject_entity.append(new_subject_ent)
        object_entity.append(new_object_ent)
  
  return sentences, subject_entity, object_entity, special_token_list


def preprocessing_dataset(dataset, preprocessing_mode) :  
  special_token_list = []
  subject_entity = []
  object_entity = []
    
  if preprocessing_mode == 'default' :
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      i = i['word'] # 수정된 부분
      j = j['word'] # 수정된 부분
      subject_entity.append(i)
      object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                                'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
   
  elif preprocessing_mode == 'punct_kr' :
    sentences, subject_entity, object_entity =  processing_special_punct_kr_tokens(dataset)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,
                                'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    
  elif preprocessing_mode == 'entity_masking' :
    sentences, subject_entity, object_entity, special_token_list =  processing_special_entity_masking_tokens(dataset)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,
                                'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    
  elif preprocessing_mode == 'punct_eng' :
    sentences, subject_entity, object_entity =  processing_special_punct_eng_tokens(dataset)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences,
                                'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    
  else : 
    assert False, "해당하는 전처리 mode가 없습니다."
  
  print(out_dataset['sentence'].head(2))
  return out_dataset, special_token_list

# style: add semantic typing
def semantic_typing(dataset, punct_mode, sentence_mode) :
  print(f"punct_mode: {punct_mode}, sentence_mode: {sentence_mode}")
  type_mapping = {'PER': '사람', 'ORG': '기관', 'DAT': '날짜', 'LOC': '위치', 'POH': '기타', 'NOH': '수량'}
  semantic_sentence = []
  
  if punct_mode == 'punct_kr' :
    
    if sentence_mode == "1":
      for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        sentence = f'{e01["word"]}와 {e02["word"]}의 관계는 {type_mapping[e01["type"]]}과 {type_mapping[e02["type"]]}의 관계이다.' 
        semantic_sentence.append(sentence)
    elif sentence_mode == "2":
      for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        sentence = f'{type_mapping[e01["type"]]} {e01["word"]}와 {type_mapping[e02["type"]]} {e02["word"]}의 관계에 대해 설명하시오.'
        semantic_sentence.append(sentence)
    else : assert False, "해당하는 sentence mode가 없습니다."
    
  elif punct_mode == 'punct_eng':
    
    if sentence_mode == "1":
      for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        sentence = f'{e01["word"]}와 {e02["word"]}의 관계는 {e01["type"]}와 {e02["type"]}의 관계이다.'
        semantic_sentence.append(sentence)
    elif sentence_mode == "2":
      for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        sentence = f'{e01["type"]} {e01["word"]}와 {e02["type"]} {e02["word"]}의 관계에 대해 설명하시오.' 
        semantic_sentence.append(sentence)
    else:
      assert False, "해당하는 sentence mode가 없습니다."
      
  else : assert False, "해당하는 preprocessing/punctuation mode가 없습니다."
  return semantic_sentence
        
