"""
Sequential 추천 모델을 위한 데이터 전처리 (BERT4Rec, SASRec)
- 시간 순서 보존
- User별 시퀀스 생성
- Temporal split (Leave-one-out)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pickle
from tqdm import tqdm

class SequentialPreprocessor:
    """Sequential Recommendation을 위한 전처리"""
    
    def __init__(self, data_dir=".", output_dir="data/processed_sequential"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.movies = []
        self.peoples = {}
        self.ratings = []
        
    def load_ndjson(self, filename):
        """NDJSON 파일 로드"""
        file_path = self.data_dir / filename
        print(f"Loading {filename}...")
        
        if filename == 'ratings.ndjson':
            chunks = []
            for chunk in tqdm(pd.read_json(file_path, lines=True, chunksize=100000)):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            return df.to_dict('records')
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    data.append(json.loads(line))
                except:
                    pass
        return data
    
    def load_all_data(self):
        """모든 데이터 로드"""
        print("=" * 70)
        print("Step 1: Loading Raw Data")
        print("=" * 70)
        
        self.movies = self.load_ndjson('movies.ndjson')
        print(f"✓ Loaded {len(self.movies):,} movies")
        
        peoples_list = self.load_ndjson('peoples.ndjson')
        self.peoples = {p['_id']: p for p in peoples_list}
        print(f"✓ Loaded {len(self.peoples):,} people")
        
        self.ratings = self.load_ndjson('ratings.ndjson')
        print(f"✓ Loaded {len(self.ratings):,} ratings")
    
    def filter_data(self, min_user_interactions=5, min_movie_interactions=3, 
                    min_rating_threshold=7):
        """데이터 필터링 (기존과 동일)"""
        print("\n" + "=" * 70)
        print("Step 2: Filtering (Rating 7-10, Min Interactions)")
        print("=" * 70)
        
        df = pd.DataFrame(self.ratings)
        print(f"Original: {len(df):,} ratings")
        
        # Rating 필터링
        df = df[df['rate'] > 0]
        df = df[df['rate'] >= min_rating_threshold]
        print(f"After rating >= {min_rating_threshold}: {len(df):,}")
        
        # 비정상 사용자 제거
        user_stats = df.groupby('user').agg({
            'rate': ['count', lambda x: (x == 10).sum()]
        })
        user_stats.columns = ['total', 'rating_10_count']
        user_stats['rating_10_ratio'] = user_stats['rating_10_count'] / user_stats['total']
        abnormal_users = user_stats[user_stats['rating_10_ratio'] > 0.7].index
        df = df[~df['user'].isin(abnormal_users)]
        print(f"After removing abnormal users: {len(df):,}")
        
        # 반복 필터링
        for iteration in range(5):
            user_counts = df['user'].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            df = df[df['user'].isin(valid_users)]
            
            movie_counts = df['movie'].value_counts()
            valid_movies = movie_counts[movie_counts >= min_movie_interactions].index
            df = df[df['movie'].isin(valid_movies)]
            
            print(f"Iteration {iteration + 1}: {len(df):,} ratings, "
                  f"{df['user'].nunique():,} users, {df['movie'].nunique():,} movies")
        
        self.ratings_filtered = df
        return df
    
    def create_mappings(self, df):
        """User/Movie ID 매핑"""
        print("\n" + "=" * 70)
        print("Step 3: Creating ID Mappings")
        print("=" * 70)
        
        unique_users = sorted(df['user'].unique())
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        
        unique_movies = sorted(df['movie'].unique())
        self.movie2idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx2movie = {idx: movie for movie, idx in self.movie2idx.items()}
        
        self.movie_dict = {m['_id']: m for m in self.movies}
        
        print(f"✓ Users: {len(self.user2idx):,}")
        print(f"✓ Movies: {len(self.movie2idx):,}")
        
        df['user_idx'] = df['user'].map(self.user2idx)
        df['movie_idx'] = df['movie'].map(self.movie2idx)
        
        return df
    
    def create_sequences(self, df, max_seq_length=50):
        """
        User별 시퀀스 생성 (Sequential 모델의 핵심!)
        
        Args:
            max_seq_length: 최대 시퀀스 길이 (Transformer 메모리 한계)
        """
        print("\n" + "=" * 70)
        print("Step 4: Creating User Sequences (TEMPORAL ORDER)")
        print("=" * 70)
        
        # 시간 순서로 정렬 (가장 중요!)
        df = df.sort_values(['user_idx', 'time']).reset_index(drop=True)
        
        # User별 시퀀스 생성
        user_sequences = {}
        user_timestamps = {}
        user_ratings = {}
        
        for user_idx in tqdm(df['user_idx'].unique(), desc="Creating sequences"):
            user_df = df[df['user_idx'] == user_idx].sort_values('time')
            
            # 시퀀스
            seq = user_df['movie_idx'].tolist()
            times = user_df['time'].tolist()
            rates = user_df['rate'].tolist()
            
            # 최대 길이 제한 (최근 것만)
            if len(seq) > max_seq_length:
                seq = seq[-max_seq_length:]
                times = times[-max_seq_length:]
                rates = rates[-max_seq_length:]
            
            user_sequences[user_idx] = seq
            user_timestamps[user_idx] = times
            user_ratings[user_idx] = rates
        
        print(f"✓ Created {len(user_sequences):,} user sequences")
        
        # 통계
        seq_lengths = [len(seq) for seq in user_sequences.values()]
        print(f"\nSequence length statistics:")
        print(f"  Min: {np.min(seq_lengths)}")
        print(f"  Max: {np.max(seq_lengths)}")
        print(f"  Mean: {np.mean(seq_lengths):.1f}")
        print(f"  Median: {np.median(seq_lengths):.1f}")
        print(f"  >= {max_seq_length}: {sum(1 for l in seq_lengths if l >= max_seq_length)}")
        
        self.user_sequences = user_sequences
        self.user_timestamps = user_timestamps
        self.user_ratings = user_ratings
        self.max_seq_length = max_seq_length
        
        return user_sequences
    
    def temporal_split(self, split_type='leave_one_out'):
        """
        시간 기반 Train/Valid/Test 분리
        
        split_type:
          - 'leave_one_out': 마지막 1개 test, 마지막에서 2번째 valid
          - 'leave_k_out': 마지막 k개 test (더 robust)
        """
        print("\n" + "=" * 70)
        print("Step 5: Temporal Split (Leave-One-Out)")
        print("=" * 70)
        
        train_sequences = {}
        valid_sequences = {}
        test_sequences = {}
        
        train_data = []
        valid_data = []
        test_data = []
        
        for user_idx, seq in tqdm(self.user_sequences.items(), desc="Splitting"):
            seq_len = len(seq)
            
            if seq_len < 3:
                # 너무 짧은 시퀀스는 train만
                train_sequences[user_idx] = seq
                continue
            
            # Leave-one-out split
            train_seq = seq[:-2]     # 처음부터 n-2까지
            valid_item = seq[-2]     # 마지막에서 2번째
            test_item = seq[-1]      # 마지막
            
            train_sequences[user_idx] = train_seq
            valid_sequences[user_idx] = valid_item
            test_sequences[user_idx] = test_item
            
            # DataFrame 형태로도 저장 (평가용)
            # Train: 시퀀스의 각 아이템
            for item in train_seq:
                train_data.append({
                    'user_idx': user_idx,
                    'movie_idx': item,
                })
            
            # Valid: (user, target_item)
            valid_data.append({
                'user_idx': user_idx,
                'movie_idx': valid_item,
            })
            
            # Test: (user, target_item)
            test_data.append({
                'user_idx': user_idx,
                'movie_idx': test_item,
            })
        
        train_df = pd.DataFrame(train_data)
        valid_df = pd.DataFrame(valid_data)
        test_df = pd.DataFrame(test_data)
        
        print(f"\n✅ Temporal split completed:")
        print(f"  Train sequences: {len(train_sequences):,}")
        print(f"  Train items: {len(train_df):,}")
        print(f"  Valid items: {len(valid_df):,}")
        print(f"  Test items: {len(test_df):,}")
        
        print(f"\n⭐ Key difference from Random split:")
        print(f"  - Train: 과거 시청 이력")
        print(f"  - Valid: 마지막에서 2번째 영화")
        print(f"  - Test: 마지막 영화")
        print(f"  - 시간 순서 보존 → 더 현실적인 평가!")
        
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        
        return train_df, valid_df, test_df, train_sequences, valid_sequences, test_sequences
    
    def save_sequential_data(self, train_df, valid_df, test_df):
        """Sequential 데이터 저장"""
        print("\n" + "=" * 70)
        print("Step 6: Saving Sequential Data")
        print("=" * 70)
        
        # 1. DataFrame 저장 (평가용)
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        valid_df.to_csv(self.output_dir / 'valid.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        print(f"✓ Saved DataFrames")
        
        # 2. Sequences 저장 (학습용) ⭐ 가장 중요!
        sequences_data = {
            'train_sequences': self.train_sequences,
            'valid_sequences': self.valid_sequences,
            'test_sequences': self.test_sequences,
            'user_sequences': self.user_sequences,  # 전체 시퀀스
            'max_seq_length': self.max_seq_length,
        }
        
        with open(self.output_dir / 'sequences.pkl', 'wb') as f:
            pickle.dump(sequences_data, f)
        print(f"✓ Saved sequences.pkl (for BERT4Rec/SASRec)")
        
        # 3. Mappings 저장
        mappings = {
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'movie2idx': self.movie2idx,
            'idx2movie': self.idx2movie,
            'num_users': len(self.user2idx),
            'num_movies': len(self.movie2idx),
            'num_items': len(self.movie2idx) + 1,  # +1 for MASK token
            'mask_token': len(self.movie2idx),     # Special token
        }
        
        with open(self.output_dir / 'mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        print(f"✓ Saved mappings.pkl")
        
        # 4. Movie info 저장 (기존과 동일)
        movie_info = {}
        for idx in range(len(self.movie2idx)):
            movie_id = self.idx2movie[idx]
            movie = self.movie_dict.get(movie_id, {})
            
            genres = []
            genres_str = movie.get('genres', [])
            if genres_str:
                try:
                    genres = eval(', '.join(genres_str))
                except:
                    pass
            
            actors = []
            for actor_id in movie.get('main_cast_people_ids', [])[:5]:
                person = self.peoples.get(actor_id, {})
                name = person.get('korean') or person.get('original', '')
                if name:
                    actors.append(name)
            
            movie_info[movie_id] = {
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
        print(f"✓ Saved movie_info.pkl")
        
        print(f"\n{'='*70}")
        print(f"✅ All sequential data saved to {self.output_dir}")
        print(f"{'='*70}")
        
        # 통계 출력
        print(f"\n📊 Final Statistics:")
        print(f"  Users: {len(self.user2idx):,}")
        print(f"  Movies: {len(self.movie2idx):,}")
        print(f"  Train sequences: {len(self.train_sequences):,}")
        print(f"  Valid items: {len(valid_df):,}")
        print(f"  Test items: {len(test_df):,}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  MASK token ID: {len(self.movie2idx)}")


def main():
    """전체 전처리 파이프라인"""
    
    preprocessor = SequentialPreprocessor()
    
    # 1. 데이터 로드
    preprocessor.load_all_data()
    
    # 2. 필터링 (기존과 동일)
    df = preprocessor.filter_data(
        min_user_interactions=5,
        min_movie_interactions=3,
        min_rating_threshold=7
    )
    
    # 3. ID 매핑
    df = preprocessor.create_mappings(df)
    
    # 4. 시퀀스 생성 (⭐ 새로운 부분!)
    preprocessor.create_sequences(df, max_seq_length=50)
    
    # 5. Temporal split (⭐ 새로운 부분!)
    train_df, valid_df, test_df, *_ = preprocessor.temporal_split()
    
    # 6. 저장
    preprocessor.save_sequential_data(train_df, valid_df, test_df)
    
    print("\n" + "=" * 70)
    print("✅ Sequential 전처리 완료!")
    print("=" * 70)
    print("\n📝 다음 단계:")
    print("  1. data/processed_sequential/ 폴더 확인")
    print("  2. sequences.pkl 파일 확인 (BERT4Rec/SASRec용)")
    print("  3. 2_train_bert4rec.py 또는 2_train_sasrec.py 실행")
    print("\n💡 주요 차이점:")
    print("  - 시간 순서 보존 (Temporal order)")
    print("  - User별 시퀀스 생성")
    print("  - Leave-one-out split")
    print("  - 더 현실적인 평가!")


if __name__ == "__main__":
    main()

