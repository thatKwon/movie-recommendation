# 영화 추천 시스템

**목표**: NDCG@10 > 0.4  
**환경**: Google Colab A100 GPU  
**데이터**: NDJSON 형식 (movies, peoples, ratings)

---

## 📁 프로젝트 구조

```
deeplearning/
├── ⚙️ 데이터 전처리
│   ├── 1_data_preprocessing.py              # 딥러닝 추천 모델용 (LightGCN, BPR-MF)
│   └── 1_data_preprocessing_sequential.py   # Sequential 모델용 (BERT4Rec)
│
├── 🧠 딥러닝 모델 학습
│   ├── 2_train_lightgcn.py     # Graph Neural Network 기반
│   ├── 2_train_bpr_mf.py       # Neural Matrix Factorization
│   └── 2_train_bert4rec.py     # Transformer 기반
│
└── 🎯 추론
    └── 4_inference.py          # Top-50 추천 생성
```

---

## 🚀 실행 방법

### Step 1: 데이터 전처리

#### 방법 A: Graph/Matrix 기반 딥러닝 모델 (LightGCN, BPR-MF)

**처리 과정**:
- NDJSON 파일 로드 (movies, peoples, ratings)
- User-Item 상호작용 그래프 생성
- Rating 기반 Positive/Negative 분리
- Train/Valid/Test 분할 (8:1:1)
- COO 형식 sparse matrix 생성

**출력 파일** (`data/processed/`):
- `mappings.pkl`: User/Movie ID 매핑
- `train_interactions.pkl`: 학습 데이터
- `valid_interactions.pkl`: 검증 데이터
- `test_interactions.pkl`: 테스트 데이터
- `movie_info.pkl`: 영화 메타데이터

---

#### 방법 B: Sequential 모델 (BERT4Rec)

**처리 과정**:
- 시간 순서 보존 (timestamp 기준 정렬)
- User별 시퀀스 생성
- Temporal split (Leave-one-out)
- 최소 상호작용 수 필터링 (min_interactions ≥ 5)

**출력 파일** (`data/processed_sequential/`):
- `mappings.pkl`: User/Movie ID 매핑
- `train_sequences.pkl`: User별 학습 시퀀스
- `valid_sequences.pkl`: 검증 데이터
- `test_sequences.pkl`: 테스트 데이터

---

### Step 2: 모델 학습

#### 모델 1: LightGCN

**모델 특징**:
- Graph Neural Network (GNN) 기반 딥러닝 추천 모델
- User-Item 상호작용 그래프에서 embedding 전파
- 협업 필터링을 GNN으로 구현
- He et al., SIGIR 2020

**주요 하이퍼파라미터**:
```python
embedding_dim = 64      # 임베딩 차원
num_layers = 3          # GNN 레이어 수
lr = 0.001             # Learning rate
batch_size = 16384      # 배치 크기
reg_weight = 1e-4       # L2 정규화
```

**학습 시간**: 15-20분 (A100)  
**예상 NDCG@10**: 0.20-0.40

---

#### 모델 2: BPR Matrix Factorization (Neural MF)

**모델 특징**:
- PyTorch로 구현한 Neural Matrix Factorization
- Bayesian Personalized Ranking (BPR) Loss 사용
- 간단하지만 강력한 딥러닝 baseline

**주요 하이퍼파라미터**:
```python
embedding_dim = 64      # 임베딩 차원
lr = 0.001             # Learning rate
batch_size = 16384      # 배치 크기
reg_weight = 1e-4       # L2 정규화
```

**학습 시간**: 10-15분 (A100)  
**예상 NDCG@10**: 0.15-0.35

---

#### 모델 3: BERT4Rec (Sequential)

**모델 특징**:
- Transformer 기반 Sequential 추천
- Bidirectional self-attention
- Masked item prediction
- Sun et al., CIKM 2019

**주요 하이퍼파라미터**:
```python
d_model = 128           # 임베딩 차원
num_heads = 4           # Attention heads
num_layers = 2          # Transformer layers
max_seq_len = 50        # 최대 시퀀스 길이
mask_prob = 0.15        # Masking 확률
```

**학습 시간**: 20-30분 (A100)  
**예상 NDCG@10**: 0.25-0.45

---

### Step 3: 추론 및 추천 생성

**기능**:
- 학습된 모델로 Top-50 영화 추천
- User별 추천 결과 생성
- 이미 평가한 영화 제외
- 추천 이유 분석 (선택)

**출력 파일**:
- `recommendations_top50.json`: 최종 추천 결과

---

## 📊 데이터 전처리 상세

### Graph/Matrix 기반 모델 전처리 (`1_data_preprocessing.py`)

**주요 기능**:

1. **데이터 로드**
   - NDJSON 파일을 chunking으로 메모리 효율적 로드
   - Movies: 영화 정보 (제목, 장르, 감독, 배우 등)
   - Ratings: User-Movie-Rating-Timestamp

2. **필터링**
   - 최소 상호작용 수 필터링
   - User: min 5개 이상
   - Movie: min 10개 이상

3. **Positive/Negative 정의**
   - Positive: Rating ≥ 7
   - Negative: Rating < 7
   - (또는 Explicit Negative 전략 사용 가능)

4. **데이터 분할**
   - Train: 80%
   - Valid: 10%
   - Test: 10%
   - Random split 또는 Temporal split

5. **출력 형식**
   - COO (Coordinate) format sparse matrix
   - User ID, Item ID 매핑 테이블

---

### Sequential 전처리 (`1_data_preprocessing_sequential.py`)

**주요 기능**:

1. **시간 순서 보존**
   - Timestamp 기준 정렬
   - User별 평가 시퀀스 생성

2. **시퀀스 생성**
   - 각 User의 전체 평가 이력을 시퀀스로 변환
   - 최소 시퀀스 길이 필터링 (min 5개)

3. **Temporal Split**
   - Train: 시간상 처음 ~ (n-2)번째
   - Valid: (n-1)번째
   - Test: n번째 (가장 최근)

4. **출력 형식**
   - User별 시퀀스 리스트
   - Item ID는 1부터 시작 (0은 padding/mask)

---

## 🧠 모델 학습 상세

### LightGCN

**알고리즘**:
```
1. User와 Item의 초기 embedding을 Neural Network로 학습
2. Graph Convolution으로 embedding 전파
   - Layer 0: 초기 embedding
   - Layer k: 이웃 노드들의 평균 (Message Passing)
3. 모든 레이어의 embedding을 평균하여 최종 embedding 생성
4. BPR Loss로 학습 (Pairwise Ranking)
```

**손실 함수**:
- BPR Loss: user가 positive item을 negative item보다 선호하도록 학습
- L2 Regularization: Overfitting 방지

**평가 지표**:
- NDCG@10: Top-10 추천의 순위 품질 (정규화된 DCG)
- Recall@10: Top-10에 정답이 포함된 비율
- HR@10: Hit Rate (정답 포함 여부)

---

### BPR Matrix Factorization (Neural MF)

**알고리즘**:
```
1. User embedding과 Item embedding을 Neural Network로 학습
2. User bias와 Item bias 추가 (선형 보정)
3. Score = user·item + user_bias + item_bias + global_bias
4. BPR Loss로 학습 (Pairwise Ranking)
```

**특징**:
- Neural Network로 구현한 Matrix Factorization
- LightGCN보다 단순하지만 효과적
- Graph 구조 없이 단순 embedding으로 학습
- 빠른 학습 및 추론

---

### BERT4Rec

**알고리즘**:
```
1. User의 시퀀스를 입력으로 받음
2. 랜덤하게 일부 item을 [MASK]로 치환
3. Transformer로 양방향 context 학습
4. Masked item 예측
5. Cross Entropy Loss로 학습
```

**특징**:
- 양방향 모델링 (이전 + 이후 context)
- Sequential 패턴 학습
- Cold-start 문제에 강함

---

## 🎯 하이퍼파라미터 튜닝 가이드

### LightGCN

```python
# 성능 우선 (메모리 충분)
embedding_dim = 128
num_layers = 3
batch_size = 16384
lr = 0.001

# 속도 우선 (메모리 부족)
embedding_dim = 64
num_layers = 2
batch_size = 32768
lr = 0.001

# Overfitting 발생 시
reg_weight = 5e-4    # 1e-4 → 5e-4
patience = 3         # Early stopping
lr = 0.0005          # Learning rate 감소
```

---

### BPR-MF

```python
# 기본 설정
embedding_dim = 64
lr = 0.001
batch_size = 16384
reg_weight = 1e-4

# 성능 개선
embedding_dim = 128  # 더 큰 임베딩
lr = 0.0005         # 더 작은 learning rate
```

---

### BERT4Rec

```python
# 기본 설정
d_model = 128
num_heads = 4
num_layers = 2
max_seq_len = 50
mask_prob = 0.15

# 성능 우선
d_model = 256        # 더 큰 모델
num_layers = 4       # 더 깊은 모델

# 속도 우선
d_model = 64         # 더 작은 모델
num_layers = 1       # 얕은 모델
batch_size = 512     # 큰 배치
```

---

## 📈 성공 기준

```
✅ NDCG@10 ≥ 0.40: Excellent
✅ NDCG@10 ≥ 0.30: Good
⚠️  NDCG@10 ≥ 0.20: Acceptable
❌ NDCG@10 < 0.20: Need Improvement
```

**최종 수정**: 2025-11-04  
**버전**: v2.0
