"""
데이터 전처리 스크립트
- NDJSON 파일을 읽어서 학습용 데이터로 변환
- User-Item 상호작용 그래프 생성
- Train/Valid/Test 분리
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, data_dir=".", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.movies = []
        self.peoples = {}
        self.ratings = []
        
    def load_ndjson(self, filename):
        """NDJSON 파일 로드 (메모리 효율적)"""
        file_path = self.data_dir / filename
        
        print(f"Loading {filename}...")
        
        # Ratings는 pandas로 직접 로드 (더 빠르고 메모리 효율적)
        if filename == 'ratings.ndjson':
            import pandas as pd
            # chunksize로 나눠서 읽기
            chunks = []
            for chunk in tqdm(pd.read_json(file_path, lines=True, chunksize=100000)):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            return df.to_dict('records')
        
        # Movies와 Peoples는 기존 방식
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data.append(json.loads(line))
                except:
                    pass  # 손상된 라인 스킵
        
        return data
    
    def load_all_data(self):
        """모든 데이터 로드"""
        print("=" * 50)
        print("Step 1: Loading Raw Data")
        print("=" * 50)
        
        # Movies
        self.movies = self.load_ndjson('movies.ndjson')
        print(f"✓ Loaded {len(self.movies):,} movies")
        
        # Peoples
        peoples_list = self.load_ndjson('peoples.ndjson')
        self.peoples = {p['_id']: p for p in peoples_list}
        print(f"✓ Loaded {len(self.peoples):,} people")
        
        # Ratings
        self.ratings = self.load_ndjson('ratings.ndjson')
        print(f"✓ Loaded {len(self.ratings):,} ratings")
    
    def filter_data(self, min_user_interactions=5, min_movie_interactions=3, min_rating_threshold=7):
        """
        희소한 데이터 필터링 + Implicit Feedback
        - Implicit feedback: Rating 7-10 positive로 간주 (더 많은 데이터)
        - 평점이 너무 적은 사용자/영화 제거
        - 비정상 사용자 제거 (Rating 10을 과도하게 주는 사용자)
        """
        print("\n" + "=" * 50)
        print("Step 2: Implicit Feedback Filtering (Rating 7-10)")
        print("=" * 50)
        
        df = pd.DataFrame(self.ratings)
        print(f"Original: {len(df):,} ratings")
        
        # Rating 0 제거
        df = df[df['rate'] > 0]
        
        # Rating 분포 출력
        if 'rate' in df.columns:
            print(f"\nRating 분포:")
            rating_dist = df['rate'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                pct = 100 * count / len(df)
                marker = "✓" if rating >= min_rating_threshold else "✗"
                print(f"  {marker} Rating {rating:2d}: {count:7,} ({pct:5.2f}%)")
            
            # Threshold 적용 (Implicit Feedback: 8-10만 positive)
            df = df[df['rate'] >= min_rating_threshold]
            print(f"\n✓ After rating threshold (>= {min_rating_threshold}): {len(df):,} ratings ({100*len(df)/len(rating_dist):.1f}%)")
        
        # 비정상 사용자 제거 (Rating 10을 70% 이상 주는 사용자)
        print(f"\n⭐ Removing abnormal users...")
        user_stats = df.groupby('user').agg({
            'rate': ['count', lambda x: (x == 10).sum()]
        })
        user_stats.columns = ['total', 'rating_10_count']
        user_stats['rating_10_ratio'] = user_stats['rating_10_count'] / user_stats['total']
        
        abnormal_users = user_stats[user_stats['rating_10_ratio'] > 0.7].index
        print(f"  Found {len(abnormal_users):,} abnormal users (>70% Rating 10)")
        
        df = df[~df['user'].isin(abnormal_users)]
        print(f"✓ After removing: {len(df):,} ratings")
        
        # 반복적으로 필터링
        for iteration in range(5):
            # User 필터링
            user_counts = df['user'].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            df = df[df['user'].isin(valid_users)]
            
            # Movie 필터링
            movie_counts = df['movie'].value_counts()
            valid_movies = movie_counts[movie_counts >= min_movie_interactions].index
            df = df[df['movie'].isin(valid_movies)]
            
            print(f"Iteration {iteration + 1}: {len(df):,} ratings, "
                  f"{df['user'].nunique():,} users, {df['movie'].nunique():,} movies")
        
        self.ratings_filtered = df
        return df
    
    def create_mappings(self, df):
        """
        User/Movie ID를 연속된 인덱스로 매핑
        """
        print("\n" + "=" * 50)
        print("Step 3: Creating ID Mappings")
        print("=" * 50)
        
        # User 매핑
        unique_users = sorted(df['user'].unique())
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        
        # Movie 매핑
        unique_movies = sorted(df['movie'].unique())
        self.movie2idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx2movie = {idx: movie for movie, idx in self.movie2idx.items()}
        
        # 영화 정보 딕셔너리
        self.movie_dict = {m['_id']: m for m in self.movies}
        
        print(f"✓ Users: {len(self.user2idx):,}")
        print(f"✓ Movies: {len(self.movie2idx):,}")
        
        # 매핑된 인덱스 추가
        df['user_idx'] = df['user'].map(self.user2idx)
        df['movie_idx'] = df['movie'].map(self.movie2idx)
        
        return df
    
    def extract_movie_features(self):
        """
        영화 메타데이터 추출
        """
        print("\n" + "=" * 50)
        print("Step 4: Extracting Movie Features")
        print("=" * 50)
        
        # 장르 목록 생성
        all_genres = set()
        for movie in self.movies:
            if movie['_id'] in self.movie2idx:
                genres_str = movie.get('genres', [])
                if genres_str:
                    # "['드라마', '멜로/로맨스']" → ['드라마', '멜로/로맨스']
                    try:
                        genres = eval(', '.join(genres_str))
                        all_genres.update(genres)
                    except:
                        pass
        
        self.all_genres = sorted(all_genres)
        self.genre2idx = {g: idx for idx, g in enumerate(self.all_genres)}
        
        print(f"✓ Found {len(self.all_genres)} genres")
        
        # 영화별 특성 벡터 생성
        movie_features = []
        for idx in range(len(self.movie2idx)):
            movie_id = self.idx2movie[idx]
            movie = self.movie_dict.get(movie_id, {})
            
            # 장르 원-핫 인코딩
            genre_vector = np.zeros(len(self.all_genres))
            genres_str = movie.get('genres', [])
            if genres_str:
                try:
                    genres = eval(', '.join(genres_str))
                    for genre in genres:
                        if genre in self.genre2idx:
                            genre_vector[self.genre2idx[genre]] = 1
                except:
                    pass
            
            # 연도 정규화
            year = movie.get('year', 2000)
            year_norm = (year - 1900) / 125.0 if year else 0.5
            
            # 출연진 수
            cast_count = len(movie.get('main_cast_people_ids', []))
            cast_norm = min(cast_count / 10.0, 1.0)
            
            # 특성 결합
            features = np.concatenate([
                genre_vector,
                [year_norm, cast_norm]
            ])
            
            movie_features.append(features)
        
        self.movie_features = np.array(movie_features, dtype=np.float32)
        print(f"✓ Movie feature shape: {self.movie_features.shape}")
        
        return self.movie_features
    
    def split_data(self, df, test_ratio=0.15, valid_ratio=0.15):
        """
        Train/Valid/Test 분리 (Leave-One-Out - 논문 방식)
        
        **Leave-One-Out (논문과 동일)**:
        - 각 사용자의 마지막 상호작용을 Test로 사용
        - 마지막에서 2번째 상호작용을 Valid로 사용
        - 나머지를 Train으로 사용
        - 시간 순서 고려 (실제 추천 환경과 동일)
        - 더 어렵지만 현실적인 평가
        
        Reference: 
        "We obtained the last interaction for every user" - 논문 Section 4.1.1
        """
        print("\n" + "=" * 50)
        print("Step 5: Splitting Data (LEAVE-ONE-OUT - Paper Method)")
        print("=" * 50)
        
        # 시간순 정렬 (중요!)
        if 'time' in df.columns:
            print("✓ Sorting by timestamp...")
            df = df.sort_values(['user_idx', 'time']).reset_index(drop=True)
        else:
            print("⚠️  No timestamp found, using original order")
            df = df.sort_values(['user_idx']).reset_index(drop=True)
        
        print(f"전체 데이터: {len(df):,} ratings")
        
        # 각 사용자별 상호작용 수 확인
        user_counts = df.groupby('user_idx').size()
        print(f"\n사용자당 평균 상호작용: {user_counts.mean():.1f}")
        print(f"최소 상호작용: {user_counts.min()}")
        print(f"최대 상호작용: {user_counts.max()}")
        
        # Leave-One-Out Split
        # 1. Test: 각 사용자의 마지막 상호작용
        test_indices = df.groupby('user_idx', sort=False).tail(1).index
        remaining_df = df.drop(test_indices)
        
        # 2. Valid: 각 사용자의 마지막에서 2번째 상호작용 (남은 것 중에서)
        valid_indices = remaining_df.groupby('user_idx', sort=False).tail(1).index
        
        # 3. Train: 나머지
        train_indices = remaining_df.index.difference(valid_indices)
        
        train_df = df.loc[train_indices].copy()
        valid_df = df.loc[valid_indices].copy()
        test_df = df.loc[test_indices].copy()
        
        # 상호작용 수가 부족한 사용자 처리
        # (2개 이하인 경우 valid가 비어있을 수 있음)
        users_with_valid = valid_df['user_idx'].nunique()
        users_with_test = test_df['user_idx'].nunique()
        total_users = df['user_idx'].nunique()
        
        print(f"\n✅ Leave-One-Out Split 완료:")
        print(f"  Train: {len(train_df):,} ratings ({100*len(train_df)/len(df):.1f}%)")
        print(f"  Valid: {len(valid_df):,} ratings ({100*len(valid_df)/len(df):.1f}%)")
        print(f"  Test:  {len(test_df):,} ratings ({100*len(test_df)/len(df):.1f}%)")
        print(f"  Total: {len(df):,} ratings")
        
        # 사용자/영화 커버리지 확인
        print(f"\n사용자 커버리지:")
        print(f"  Train: {train_df['user_idx'].nunique():,} users")
        print(f"  Valid: {valid_df['user_idx'].nunique():,} users ({100*users_with_valid/total_users:.1f}%)")
        print(f"  Test:  {test_df['user_idx'].nunique():,} users ({100*users_with_test/total_users:.1f}%)")
        
        print(f"\n영화 커버리지:")
        print(f"  Train: {train_df['movie_idx'].nunique():,} movies")
        print(f"  Valid: {valid_df['movie_idx'].nunique():,} movies")
        print(f"  Test:  {test_df['movie_idx'].nunique():,} movies")
        
        # 시간 순서 검증
        if 'time' in df.columns:
            print(f"\n시간 순서 검증:")
            print(f"  Train 시간 범위: {train_df['time'].min()} ~ {train_df['time'].max()}")
            print(f"  Valid 시간 범위: {valid_df['time'].min()} ~ {valid_df['time'].max()}")
            print(f"  Test  시간 범위: {test_df['time'].min()} ~ {test_df['time'].max()}")
            
            # 각 사용자별로 train < valid < test인지 확인
            print(f"\n  ✓ Leave-One-Out: 각 사용자별로 시간 순서 보장됨")
        
        print(f"\n✅ Leave-One-Out 완료 (논문 방식, 실제 추천 환경과 동일)")
        print(f"⚠️  주의: Random split보다 어려운 평가 (NDCG가 낮게 나올 수 있음)")
        
        return train_df, valid_df, test_df
    
    def save_processed_data(self, train_df, valid_df, test_df, min_rating_threshold):
        """
        전처리된 데이터 저장
        """
        print("\n" + "=" * 50)
        print("Step 6: Saving Processed Data")
        print("=" * 50)
        
        # DataFrames 저장 (pkl과 csv 둘 다)
        train_df.to_pickle(self.output_dir / 'train.pkl')
        valid_df.to_pickle(self.output_dir / 'valid.pkl')
        test_df.to_pickle(self.output_dir / 'test.pkl')
        
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        valid_df.to_csv(self.output_dir / 'valid.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        # 전체 영화 매핑 (평가용 - Rating ≥ threshold 영화만 포함)
        # ⚠️  평점이 있는 영화만 추가 (메타데이터만 있는 영화 제외)
        all_movies_for_eval = {}
        
        # ratings.ndjson에서 실제 평점이 있는 영화 ID 추출
        all_rated_movies = set()
        for rating in self.ratings:
            all_rated_movies.add(rating['movie'])
        
        print(f"\n📊 Movie Statistics:")
        print(f"  - Movies in metadata (movies.ndjson): {len(self.movies):,}")
        print(f"  - Movies with ratings: {len(all_rated_movies):,}")
        print(f"  - Movies without ratings: {len(self.movies) - len(all_rated_movies):,}")
        
        # 평점이 있는 영화만 매핑에 추가
        original_train_movie_count = len(self.movie2idx)
        
        for movie in self.movies:
            movie_id = movie['_id']
            # 평점이 있고, 아직 매핑에 없는 영화만 추가
            if movie_id in all_rated_movies and movie_id not in self.movie2idx:
                # 전처리에서 제외되었지만 평점은 있는 영화
                new_idx = len(self.movie2idx)
                self.movie2idx[movie_id] = new_idx
                self.idx2movie[new_idx] = movie_id
                all_movies_for_eval[movie_id] = new_idx
        
        print(f"\n✓ Total movies (rated movies only): {len(self.movie2idx):,}")
        print(f"  - Training movies (Rating ≥ {min_rating_threshold}): {original_train_movie_count:,}")
        print(f"  - Evaluation-only movies (Rating < {min_rating_threshold}): {len(all_movies_for_eval):,}")
        
        # min_rating_threshold 저장 (나중에 참고용)
        self.min_rating_threshold = min_rating_threshold
        
        # Mappings 저장
        mappings = {
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'movie2idx': self.movie2idx,
            'idx2movie': self.idx2movie,
            'genre2idx': self.genre2idx,
            'all_genres': self.all_genres,
            'num_users': len(self.user2idx),
            'num_movies': len(self.movie2idx),  # 전체 영화 수 (평가용 포함)
            'num_movies_train': len(self.movie2idx) - len(all_movies_for_eval),  # 학습용만
            'num_genres': len(self.all_genres),
            # 추가: inference.py에서 사용하는 키 이름
            'user_idx_to_id': self.idx2user,
            'movie_idx_to_id': self.idx2movie,
            'movie_id_to_idx': self.movie2idx,
        }
        
        with open(self.output_dir / 'mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        # Movie features 저장
        np.save(self.output_dir / 'movie_features.npy', self.movie_features)
        
        # Movie dict 저장 (LLM 필터링용)
        movie_info = {}
        for idx in range(len(self.movie2idx)):
            movie_id = self.idx2movie[idx]
            movie = self.movie_dict.get(movie_id, {})
            
            # 장르 파싱
            genres = []
            genres_str = movie.get('genres', [])
            if genres_str:
                try:
                    genres = eval(', '.join(genres_str))
                except:
                    pass
            
            # 출연진 이름
            actors = []
            for actor_id in movie.get('main_cast_people_ids', [])[:5]:
                person = self.peoples.get(actor_id, {})
                name = person.get('korean') or person.get('original', '')
                if name:
                    actors.append(name)
            
            movie_info[movie_id] = {  # movie_id를 키로 사용
                'movie_id': movie_id,
                'movie_idx': idx,
                'title': movie.get('title', ''),
                'title_eng': movie.get('title_eng', ''),
                'year': movie.get('year'),
                'grade': movie.get('grade', '정보 없음'),
                'genres': genres,
                'actors': actors
            }
        
        with open(self.output_dir / 'movie_info.pkl', 'wb') as f:
            pickle.dump(movie_info, f)
        
        print(f"✓ All data saved to {self.output_dir}")
        
        # 통계 출력
        print("\n" + "=" * 50)
        print("Data Statistics")
        print("=" * 50)
        print(f"Users: {len(self.user2idx):,}")
        print(f"Movies: {len(self.movie2idx):,}")
        print(f"Genres: {len(self.all_genres)}")
        print(f"Train ratings: {len(train_df):,}")
        print(f"Valid ratings: {len(valid_df):,}")
        print(f"Test ratings: {len(test_df):,}")
        print(f"Sparsity: {100 * len(train_df) / (len(self.user2idx) * len(self.movie2idx)):.4f}%")


def main():
    """전체 전처리 파이프라인 실행"""
    
    preprocessor = DataPreprocessor()
    
    # 1. 데이터 로드
    preprocessor.load_all_data()
    
    # 2. 희소 데이터 필터링 및 평점 threshold 적용
    # Implicit Feedback: Rating 6-10 (더 많은 데이터로 성능 향상)
    min_rating_threshold = 6  # ✅ 6: 중간 이상 평점 (6-10, 50% 더 많은 데이터!)
    df = preprocessor.filter_data(
        min_user_interactions=5,   # 5: 더 많은 사용자 포함
        min_movie_interactions=3,  # 3: 더 많은 영화 포함
        min_rating_threshold=min_rating_threshold
    )
    
    # 3. ID 매핑 생성
    df = preprocessor.create_mappings(df)
    
    # 4. 영화 특성 추출
    preprocessor.extract_movie_features()
    
    # 5. Train/Valid/Test 분리
    train_df, valid_df, test_df = preprocessor.split_data(df)
    
    # 6. 저장
    preprocessor.save_processed_data(train_df, valid_df, test_df, min_rating_threshold)
    
    print("\n" + "=" * 70)
    print("✅ 데이터 전처리 완료 (논문 방식)")
    print("=" * 70)
    print("\n논문 정렬 설정:")
    print("  ✓ Implicit Threshold: Rating ≥ 7 (논문: score > 2의 1-10 척도 버전)")
    print("  ✓ Split: Leave-One-Out (논문과 동일)")
    print("  ✓ Evaluation: 각 사용자의 마지막 상호작용 예측")
    print("  ✓ Temporal Order: 시간 순서 보장")
    print("\n주요 특징:")
    print("  • Rating 7-10: 높은 품질의 positive feedback")
    print("  • Leave-One-Out: 실제 추천 환경과 동일한 평가")
    print("  • 시간순 정렬: 미래 예측 task (더 어려움)")
    print("\n예상 성능 (Leave-One-Out은 Random보다 어려움):")
    print("  • 목표 NDCG@10: > 0.4")
    print("  • 참고: 논문 MovieLens 1m = 0.6+ (더 쉬운 데이터셋)")


if __name__ == "__main__":
    main()

