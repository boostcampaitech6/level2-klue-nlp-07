# NLP_07조 랩업 리포트

## 1. 프로젝트 개요
- 관계 추출(RE, Relation Extraction) : 문장 내의 두 단어(Entity)에 대한 속성과 관계를 예측하는 NLP Task
- KLUE-RE 데이터셋
```
# task 예시
sentence : <Something>는 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다.
subject_entity : 비틀즈, ORG (subject word, subject type)
object_entity : 조지 해리슨, PER (object word, object type)
label : no_relation
-> 주어진 문정으로 판단했을 때, ORG 타입인 '비틀즈'와 PER 타입인 '조지 해리슨'은 'no_relation' 관계이다.
```


## 2. 프로젝트 팀 구성 및 역할
|공통|모델 성능 테스트, 하이퍼파라미터 탐색, fine-tuning|
|---|---|
|양서현|데이터 클리닝, 데이터 전처리(semantic typing) wandb sweep을 통한 하이퍼파라미터 탐색|
|이상경|데이터 분할, 하이퍼파라미터 탐색|
|이승백|데이터 분석 및 데이터 샘플링|
|이주용|데이터 분석, 데이터 샘플링, 앙상블, 모델 탐색|
|정종관|데이터 분석, 데이터 클리닝, 추론 후처리|
|정지영|데이터 전처리(special token)|

## 3. 프로젝트 수행 절차 및 방법
### 데이터 분석
- Train : 32,470개 / Test : 7,765개
- Train 데이터는 가장 많은 라벨(no_relation)이 9,534개, 가장 적은 라벨(per:place_of_death)가 40개인 불균형 데이터
  ![image](https://github.com/boostcampaitech6/level2-klue-nlp-07/assets/19660039/4bceb35e-035e-40fa-abb4-2d636a414608)
- Train 데이터에서 subject_type(ORG, PER)과 label head(org:, per:)이 일치하지 않는 경우는 32,470개 중 4개임을 확인
  
### 데이터 클리닝
- sentence / subject_entity / object_entity가 전부 동일한 Train 데이터의 중복값 확인
- 라벨 값이 다를 경우, 데이터 불균형 해소를 위해 더 많은 라벨의 중복데이터를 삭제
 (예시)
|id|sentence|subject|object|label|결과|
|---|---|---|---|---|---|
|6749|대한항공은 5일 조양호 회장의 3자녀가 보유한...|대한항공, ORG|조양호, PER|no_relation|제거|
|12829|(동일)|(동일)|(동일)|org:top_members/employees|보존|

### 데이터 분할
- validation 데이터셋이 제공되지 않으므로, 보유한 train 데이터셋으로부터 검증용 데이터셋 분할
- 라벨 분포를 고려하지 않고 분할할 경우, 데이터가 부족한 특정 레이블이나 특정 타입쌍(sub_type: NOH, obj_type: *)이 검증 데이터셋에 포함되지 않는 문제 발생
![image](https://github.com/boostcampaitech6/level2-klue-nlp-07/assets/19660039/fbf72fc7-d72e-49ae-8385-05849f99d5d3)
- (sub_type, obj_type) 쌍으로 이루어진 총 12개의 쌍에 대하여 균등하게 뽑아 valid 데이터셋 확보

### 데이터 전처리
```
# 예시 데이터
sentence : <Something>는 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다.
subject_entity : 비틀즈, ORG
object_entity : 조지 해리슨, PER

# baseline model tokenizing
[CLS]비틀즈[SEP]조지 해리슨[SEP]<Something>는 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다.[SEP]
```
#### Special Token 활용

**A. Entity Marker**
- Sentence에서의 두 Entity word 대신 각각 <S-type>, <O-type>으로 Masked 하여 학습의 input format을 변경하였다. 또한, <S-type> 와 <O-type>에 해당하는 Special token을 추가하고 Embedding Layer을 추가하였다.
- e.g. 〈Something〉는 <O-PER>이 쓰고 <S-ORG>가 앨범에 담은 노래다.

**B. Typed Entity Marker with special token**
- Sentence에서 두 Entity word를 각각 [S:Type] Subject word [/S:Type],  [O:Type] Object word [/O:Type] 형태의 Special token으로 감싸주는 형태로 변경하였다.
- e.g. 〈Something〉는 [O:PER] 조지 해리슨 [/O:PER]이 쓰고 [S:ORG] 비틀즈 [/S:ORG]가 앨범에 담은 노래다.

**C. Typed Entity Marker with Punctuation**
- Special Token이나 Embedding Layer을 추가하지 않으면서 각 Entity word의 위치와 Type을 알리는 방법이다. 새로 학습시켜야하는 Special Token보다는 기학습된 “ @, *, #, ^ “ 특수문자를 이용해 Entity의 정보를 표현하는 Input Format으로 변경해준다면 모델의 성능이 향상될 것이라는 가설을 세워 실험하였다. 
- e.g. 〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 앨범에 담은 노래다.


### Semantic Typing Query
- Bert와 같은 사적 학습 모델은 **NSP**(Next Sentence Predict)의 방식으로 학습한다.
- 기존에는 [CLS]Subject Entity Word[SEP]Object Entity Word[SEP]Sentence[SEP]의 형식으로 모델에 데이터를 전달한다.
- 이 때 앞의 Entity Word를 NSP에 적합하도록, 두 단어의 관계를 설명하는 Semantic Typing으로 **Query**를 생성하여 전달한다.
```
(예시)
# 기존
[CLS]비틀즈[SEP]조지 해리슨[SEP]<Something>는 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다.[SEP]

# Semantic Query
[CLS]비틀즈와 조지 해리슨은 ORG와 PER의 관계이다.[SEP]<Something>는 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다.[SEP]
```
- Semantic Query에 Typed Entity Marker with Punctuation을 적용하면 다음과 같다.
```
# e.g.1. Semantic Query + C. Typed Entity Marker with Punctuation(ENG)
@*ORG*비틀즈@과 #^PER^조지 해리슨#는 ORG와 PER의 관계이다. [SEP]
〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 앨범에 담은 노래다.

# e.g.2. Semantic Query + C. Typed Entity Marker with Punctuation(KOR)
@*기관*비틀즈@과 #^사람^조지 해리슨#는 조직과 사람의 관계이다.  [SEP]
〈Something〉는 #^사람^조지 해리슨#이 쓰고 @*조직*비틀즈@가 앨범에 담은 노래다.

# e.g.3.  punctuation + Semantic Query (순서변경)  
〈Something〉는 #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈@가 앨범에 담은 노래다. 
[SEP] @*ORG*비틀즈@과 #^PER^조지 해리슨#는 ORG와 PER의 관계이다.  
```
- 3가지 경우, 모두 기존과는 확연한 성능 향상을 확인.그 중 **e.g.1**의 형태를 사용했을 때 가장 성능이 크게 향상되는 것을 확인.

### 모델링
**A. 모델 탐색**
- 한국어로 사전 학습된 모델들 중 RE에 많이 사용되는 모델을 각각 4 epoch 훈련
  - **klue/roberta-large** (baseline model)
  - snunlp/KR-SBERT-V40K-klueNLI-augSTS
  - snunlp/KR-BERT-char16424
  - paust/pko-t5-base, large
  - paust/pko-flan-t5-large
- LB score(리더보드 스코어) 기준으로 가장 높은 성능을 보인 **klue/roberta-large** 모델 선택

**B. 학습 성능 평가**
- Confusion Matrix를 활용한 학습 성능 평가
- ...?

**C. Hyper Paramter Tuning**
- epoch, batch, learning rate에 대해서 어떤 조합이 최적의 성능을 발휘하는지 객관적으로 평가할 수 있는 방법과 지표가 필요
- WandB sweep을 활용해 다양한 조합을 자동으로 실행하고, 시각화를 통해 직접 그 결과를 관찰
- f1 score를 기준으로 최적의 조합 탐색

 
**D. 추론 결과 후보정**
- '데이터 분석'과정에서 subject_type과 label_head사이에 상관관계가 존재함을 발견
- baseline model로 추론하였을 때, test data 7765개 중 57개의 data에서 subject type과 predicted label head가 일치하지 않는 것을 확인
- subject_type에 맞춰 predicted label head를 후보정하면 성능이 향상될 것
- 추론 결과에 대하여 subject_type과 맞지 않는 label의 확률을 조정하여 subject_type과 label_head가 일치하도록 보정한다.
  - e.g. subject_type이 ORG일 경우, 'per:'로 시작하는 label의 확률값을 0으로 조정한다.


### 앙상블
- 다양한 전처리 방식과 서로 다른 seed를 가진 모델을 통해 나온 결과를 soft voting 앙상블 수행
- 서로 다른 조건에서 나온 앙상블 결과를 다시 한 번 앙상블하여 근소한 성능 향상

## 4. 프로젝트 수행 결과
![image](https://github.com/boostcampaitech6/level2-klue-nlp-07/assets/19660039/99bfbf52-87a3-4e0a-bd23-d603b9f73709)

private micro f1 score 75.9857로 최종 순위 **1위**로 마감

## 5. 피드백
- 프로젝트 마감 후 김성현 마스터님으로부터 받은 피드백 간단하게 기술
- 데이터 분석, 클리닝, 분할, 전처리 부터 모델링, 하이퍼 파라미터 튜닝, 추론 후처리, 앙상블 까지... task 전과정을 빠짐 없이 훌륭하게 소화했다. 기업 과제였으면 좋은 점수를 받았을 것.
- 데이터를 열심히 분석한 것이 인상적. 상관관계 유추를 통해 추론 후처리를 한 것이 인상적.
- validation 데이터셋 구축방법 인상적.
  - 모델의 성능을 평가할 때, 다수의 라벨에 대해선 당연히 예측을 잘할 것. edge case에 대한 성능 중요.
  - 그렇기 때문에 실제 KLUE에서 validation 데이터셋 구축할 때 비슷한 근거로 구축하였음. edge case가 충분히 포함되도록 구축할 것.
  - 데이터 증강 시에도 비슷하게 적용. 모든 데이터보다는 edge point를 위주로.
