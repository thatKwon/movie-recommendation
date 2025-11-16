"""
추론 및 추천 생성
- 학습된 MFDNN 모델로 Top 50 영화 추천
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import importlib.util

# MFDNN 모델 import
spec = importlib.util.spec_from_file_location("train_model", "train.py")
train_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model)
MFDNN = train_model.MFDNN


class MovieRecommender:
    """영화 추천 시스템"""
    
    def __init__(self, model_path="data/models/mfdnn_best.pt", 
                 data_dir="data/processed",
                 omega_explicit=0.5):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.omega_explicit = omega_explicit
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """데이터 로드"""
        print("Loading data...")
        
        # Mappings
        with open(self.data_dir / 'mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        self.num_users = mappings['num_users']
        self.num_movies_total = mappings['num_movies']  # 전체 영화 수
        self.num_movies_train = mappings.get('num_movies_train', self.num_movies_total)  # 학습된 영화 수
        self.user_idx_to_id = mappings['user_idx_to_id']
        self.movie_idx_to_id = mappings['movie_idx_to_id']
        self.movie_id_to_idx = mappings['movie_id_to_idx']
        
        # User ID to Index 매핑 (API에서 user_id로 조회하기 위해)
        self.user_id_to_idx = {v: k for k, v in self.user_idx_to_id.items()}
        
        # Movie info
        with open(self.data_dir / 'movie_info.pkl', 'rb') as f:
            self.movie_info = pickle.load(f)
        
        # Train data (이미 본 영화 제외용)
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        # User별 이미 본 영화
        self.user_seen_movies = {}
        for user_idx in range(self.num_users):
            seen = self.train_df[self.train_df['user_idx'] == user_idx]['movie_idx'].values
            self.user_seen_movies[user_idx] = set(seen)
        
        print(f"✓ Users: {self.num_users:,}")
        print(f"✓ Movies (total): {self.num_movies_total:,}")
        if self.num_movies_train < self.num_movies_total:
            print(f"✓ Movies (trained): {self.num_movies_train:,}")
        
    def load_model(self):
        """MFDNN 모델 로드"""
        print(f"Loading MFDNN model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, 
                               weights_only=False)
        
        config = checkpoint['config']
        
        # 체크포인트의 영화 수 사용 (학습된 영화 수)
        self.num_movies = config.get('num_movies_train', config['num_movies'])
        
        # 모델 생성 (체크포인트와 동일한 구조)
        self.model = MFDNN(
            num_users=config['num_users'],
            num_items=self.num_movies,  # 체크포인트의 영화 수 사용
            embedding_dim=config['embedding_dim'],
            dnn_layers=config['dnn_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        valid_ndcg = checkpoint.get('valid_ndcg', 0)
        print(f"✓ MFDNN model loaded (NDCG: {valid_ndcg:.4f})")
        print(f"✓ Model trained on {self.num_movies:,} movies")
        print(f"✓ Omega explicit: {self.omega_explicit}")
    
    @torch.no_grad()
    def recommend_top_k(self, user_idx, k=50, return_scores=True):
        """
        사용자에게 Top-K 영화 추천
        
        Args:
            user_idx: 사용자 인덱스
            k: 추천할 영화 수
            return_scores: 점수 반환 여부
            
        Returns:
            추천 영화 리스트 (movie_idx, score)
        """
        # 학습된 영화에 대한 점수 계산 (MFDNN)
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        items_tensor = torch.arange(self.num_movies, device=self.device)  # 학습된 영화만
        
        # 배치로 점수 계산
        batch_size = 1000
        all_scores = []
        
        for start_idx in range(0, self.num_movies, batch_size):
            end_idx = min(start_idx + batch_size, self.num_movies)
            batch_items = items_tensor[start_idx:end_idx]
            batch_users = user_tensor.expand(len(batch_items))
            
            batch_scores = self.model(batch_users, batch_items, self.omega_explicit)
            all_scores.append(batch_scores.cpu())
        
        scores_trained = torch.cat(all_scores).numpy()
        
        # 전체 영화에 대한 점수 배열 생성 (학습되지 않은 영화는 -10으로 초기화)
        scores = np.full(self.num_movies_total, -10.0, dtype=np.float32)
        scores[:self.num_movies] = scores_trained  # 학습된 영화 점수 복사
        
        # 이미 본 영화 제외
        seen_movies = self.user_seen_movies.get(user_idx, set())
        scores[list(seen_movies)] = -np.inf
        
        # Top-K 추출
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        if return_scores:
            return [(int(idx), float(scores[idx])) for idx in top_k_idx]
        else:
            return [int(idx) for idx in top_k_idx]
    
    def recommend_with_info(self, user_idx, k=50):
        """
        영화 정보를 포함한 추천 (제출 형식)
        
        Returns:
            Dict with user_id and recommendations
            Format: {
                "user_id": int,
                "recommendations": [
                    {"movie_id": int, "predicted_rating": float},
                    ...
                ]
            }
        """
        recommendations = self.recommend_top_k(user_idx, k=k, return_scores=True)
        
        recommendation_list = []
        for movie_idx, score in recommendations:
            movie_id = int(self.movie_idx_to_id[movie_idx])
            
            recommendation_list.append({
                'movie_id': movie_id,
                'predicted_rating': float(score)
            })
        
        return {
            'user_id': int(self.user_idx_to_id[user_idx]),
            'recommendations': recommendation_list
        }
    
    def recommend_by_user_id(self, user_id: int, k: int = 50):
        """
        User ID로 추천 (API용)
        
        Args:
            user_id: 실제 사용자 ID
            k: 추천할 영화 수
            
        Returns:
            추천 결과 또는 None (사용자 없음)
        """
        # User ID → User Index 변환
        user_idx = self.user_id_to_idx.get(user_id)
        
        if user_idx is None:
            return None
        
        return self.recommend_with_info(user_idx, k)
    
    def get_user_stats(self, user_id: int):
        """
        사용자 통계 정보 (API용)
        
        Returns:
            사용자 통계 또는 None
        """
        user_idx = self.user_id_to_idx.get(user_id)
        
        if user_idx is None:
            return None
        
        seen_movies = self.user_seen_movies.get(user_idx, set())
        
        return {
            'user_id': user_id,
            'total_watched': len(seen_movies),
            'user_exists': True
        }
    
    def health_check(self):
        """
        시스템 상태 체크 (API용)
        """
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'model_path': str(self.model_path)
        }
    
    def generate_recommendations_for_users(self, user_indices=None, k=50, 
                                          save_dir="data/recommendations"):
        """
        여러 사용자에 대한 추천 생성 및 저장 (개별 파일)
        
        Args:
            user_indices: 추천할 사용자 인덱스 리스트 (None이면 전체)
            k: 각 사용자당 추천 영화 수
            save_dir: 저장 디렉토리
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if user_indices is None:
            user_indices = range(min(100, self.num_users))
        
        print(f"\nGenerating recommendations for {len(user_indices)} users...")
        
        for user_idx in tqdm(user_indices):
            result = self.recommend_with_info(user_idx, k=k)
            
            # JSON 저장
            output_path = save_dir / f'user_{user_idx}_top{k}.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Recommendations saved to {save_dir}/")
        
        return save_dir
    
    def generate_all_recommendations(self, k=50, output_file="data/recommendations_all.json"):
        """
        전체 사용자에 대한 추천을 하나의 JSON 파일로 생성
        
        Args:
            k: 각 사용자당 추천 영화 수
            output_file: 출력 파일 경로
        
        Returns:
            출력 파일 경로
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating recommendations for all {self.num_users} users...")
        
        all_recommendations = []
        
        for user_idx in tqdm(range(self.num_users)):
            result = self.recommend_with_info(user_idx, k=k)
            all_recommendations.append(result)
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_recommendations, f, ensure_ascii=False, indent=2)
        
        print(f"✓ All recommendations saved to {output_path}")
        print(f"✓ Total users: {len(all_recommendations)}")
        print(f"✓ Movies per user: {k}")
        
        return output_path


def main():
    """메인 함수"""
    
    print("\n" + "=" * 50)
    print("Movie Recommendation - Inference (MFDNN)")
    print("=" * 50)
    
    # Train 3 모델 (논문 방식) ⭐
    recommender = MovieRecommender(
        model_path="data/models/mfdnn_best.pt",
        data_dir="data/processed",
        omega_explicit=0.5
    )
    
    # 방법 1: 개별 파일로 저장 (샘플용 - 처음 100명)
    print("\n" + "=" * 70)
    print("Option 1: Individual Files (First 100 users)")
    print("=" * 70)
    save_dir = recommender.generate_recommendations_for_users(
        user_indices=range(min(100, recommender.num_users)),
        k=50
    )
    
    # 방법 2: 전체 사용자를 하나의 JSON 파일로 저장 (제출용)
    print("\n" + "=" * 70)
    print("Option 2: Single File (All users - for submission)")
    print("=" * 70)
    output_file = recommender.generate_all_recommendations(
        k=50,
        output_file="data/recommendations_all.json"
    )
    
    # 결과 샘플 출력
    print("\n" + "=" * 70)
    print("Sample Output Format")
    print("=" * 70)
    
    sample_path = save_dir / 'user_0_top50.json'
    with open(sample_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    
    print(f"\nUser ID: {sample['user_id']}")
    print(f"Total Recommendations: {len(sample['recommendations'])}")
    print("\nTop 10 Movies:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Movie ID':<12} {'Predicted Rating':<18}")
    print("-" * 60)
    
    for rank, movie in enumerate(sample['recommendations'][:10], 1):
        print(f"{rank:<6} {movie['movie_id']:<12} {movie['predicted_rating']:<18.4f}")
    
    print("\n" + "=" * 70)
    print("✓ MFDNN Inference completed!")
    print(f"✓ Model: MFDNN (Matrix Factorization + DNN)")
    print(f"✓ Omega explicit: {recommender.omega_explicit}")
    print(f"✓ Output format:")
    print(f"  {{")
    print(f'    "user_id": int,')
    print(f'    "recommendations": [')
    print(f'      {{"movie_id": int, "predicted_rating": float}},')
    print(f'      ...')
    print(f'    ]')
    print(f"  }}")
    print(f"\n✓ Submission file: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
