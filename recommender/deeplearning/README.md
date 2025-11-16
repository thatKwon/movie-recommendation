# 영화 추천 시스템 - MFDNN (Deep Learning)

## 프로젝트 개요

MFDNN (Matrix Factorization + Deep Neural Network) 기반 영화 추천 시스템

- **논문**: "A Top-N Movie Recommendation Framework Based on Deep Neural Network with Heterogeneous Modeling" (Applied Sciences, 2021)
- **데이터셋**: KMDB (Korean Movie Database)
- **평가 방식**: 1 positive + 99 random negatives
- **목표**: NDCG@10 > 0.4 (KMDB), HR@10 > 0.7

---

## 프로젝트 구조

```
deeplearning/
├── preprocess.py                # 데이터 전처리
├── train.py                     # MFDNN 학습
├── inference.py                 # 추론 및 추천 생성
│
├── data/
│   ├── processed/               # 전처리 데이터
│   ├── models/                  # 학습된 모델
│   ├── recommendations/         # 생성된 추천 결과
│
├── ratings.ndjson               # KMDB 평점 데이터
├── movies.ndjson                # KMDB 영화 메타데이터
└── peoples.ndjson               # KMDB 인물 메타데이터
```

---

## 빠른 시작

### Step 1: 데이터 전처리

```bash
# KMDB 데이터 전처리
python preprocess.py

```

**생성되는 파일:**
- `data/processed_kmdb/train.csv`, `valid.csv`, `test.csv`
- `data/processed_kmdb/mappings.pkl` (User/Movie ID 매핑)

---

### Step 2: 모델 학습

```bash
# MFDNN 학습
python train.py
```

**하이퍼파라미터:**
- Embedding Dimension: 64
- DNN Layers: [256, 128, 64]
- Batch Size: 1024
- Learning Rate: 0.0001
- Epochs: 50 (Early Stopping 적용, patience=7)
- Evaluation: 1 positive + 99 random negatives

**학습 결과:**
- `data/models/mfdnn_best.pt` (Best 모델)
- `data/models/history.pkl` (학습 히스토리)

**KMDB 기대 성능:**
- NDCG@10: 0.4 - 0.55
- HR@10: 0.7 - 0.8

---

### Step 3: 추론 및 추천 생성

```bash
# 모든 사용자에 대해 Top 50 영화 추천
python inference.py
```

**출력 파일:**
- `data/recommendations_all.json` (전체 사용자 추천)
- `data/recommendations/user_{user_idx}_top50.json` (개별 사용자)

**출력 형식:**
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 10001,
      "predicted_rating": 9.2456
    },
    {
      "movie_id": 10002,
      "predicted_rating": 8.9871
    }
  ]
}
```

---

## MFDNN 모델 설명

### 핵심 구조

1. **MF Component (Matrix Factorization)**
   - 명시적 피드백 (평점 7-10) 학습
   - User Embedding × Item Embedding (내적)
   - 선형 패턴 학습

2. **DNN Component (Deep Neural Network)**
   - 암묵적 피드백 (상호작용 0/1) 학습
   - User Embedding + Item Embedding (Concatenation)
   - 3층 신경망 [256 → 128 → 64 → 1]
   - 비선형 패턴 학습

3. **Bias Terms**
   - User Bias: 사용자별 평점 성향
   - Item Bias: 영화별 인기도
   - Global Bias: 전체 평균

4. **최종 점수 계산**
   ```
   score = 0.5 × MF_output + 0.5 × DNN_output + user_bias + item_bias + global_bias
   ```

### 학습 방법

- **Loss Function**: MSE (Regression) + BPR (Ranking)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau (patience=3)
- **Early Stopping**: patience=7
- **Mixed Precision Training (AMP)**: GPU 최적화

---

## 평가 방법 설명

### 1+99 평가 (논문 방식)

- **방법**: 1개 정답 영화 + 99개 랜덤 부정 샘플 = 총 100개
- **목표**: 100개 중 정답 영화의 순위 계산
- **지표**: NDCG@10, HR@10
- **특징**: 
  - 빠른 평가 속도
  - 논문 결과와 직접 비교 가능
  - 실제보다 높은 성능 나옴 (더 쉬움)

### 실제 추천은?

- 추론 시(`inference.py`)는 **모든 영화**를 대상으로 순위 계산
- 이미 본 영화는 제외
- 상위 50개 영화 추천
- 따라서 평가 방법과 실제 추천은 다름!

---

## 💡 주요 명령어

### 학습

```bash
# KMDB 데이터로 학습
python train.py

# Google Colab
!python train.py
```

### 추론

```bash
# 추천 생성
python inference.py

# Google Colab
!python inference.py
```

---

## 예상 학습 결과

### KMDB 데이터셋

| Epoch | Train Loss | Valid NDCG@10 | Valid HR@10 |
|-------|-----------|---------------|-------------|
| 1     | 0.7807    | 0.4926        | 0.7809      |
| 3     | 0.6025    | 0.5026        | 0.7869      |
| 6     | 0.5173    | 0.5098        | 0.7886      |
| **10**| **0.4389**| **0.5129** ⭐ | **0.7912** ⭐|
| 20    | 0.3263    | 0.4914        | 0.7797      |

- **Best**: Epoch 10, NDCG@10 = 0.5129, HR@10 = 0.7912
- **학습 시간**: ~30분 (GPU T4 기준, 20 epochs)

---

## 프로젝트 특징

### 장점

1. **Deep Learning 기반**: 복잡한 비선형 패턴 학습
2. **논문 재현**: 검증된 알고리즘 구현
3. **1+99 평가**: 빠른 평가 & 논문과 비교 가능
4. **Early Stopping**: 과적합 방지 & 학습 시간 단축

### 한계

1. **Cold Start**: 신규 사용자/영화는 추천 불가
2. **1+99 평가**: 실제 성능보다 과대평가 가능성
3. **메타데이터 미사용**: 현재는 User-Item 상호작용만 활용
4. **계산 비용**: GPU 필요 (CPU는 매우 느림)

---

## 참고 문헌

- **Paper**: Kim et al. (2021). "A Top-N Movie Recommendation Framework Based on Deep Neural Network with Heterogeneous Modeling". *Applied Sciences*, 11(16), 7418.
  - URL: [https://www.mdpi.com/2076-3417/11/16/7418](https://www.mdpi.com/2076-3417/11/16/7418)

---