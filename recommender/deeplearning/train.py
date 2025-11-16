"""
MFDNN: 논문 정확한 구현
- Paper: "A Top-N Movie Recommendation Framework..."
- 논문의 정확한 평가 방식 구현 (1 positive + 99 negatives)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# PyTorch AMP 호환성 처리
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
    AMP_DEVICE = 'cuda'
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        AMP_DEVICE = None
    except ImportError:
        AMP_AVAILABLE = False
        autocast = None
        GradScaler = None


class MFDNN(nn.Module):
    """MFDNN: 논문 구현"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, dnn_layers=[256, 128, 64]):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # MF Component
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        
        # DNN Component
        self.user_embedding_dnn = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_dnn = nn.Embedding(num_items, embedding_dim)
        
        # DNN Layers
        dnn_input_dim = embedding_dim * 2
        layers = []
        prev_dim = dnn_input_dim
        
        for hidden_dim in dnn_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)
        
        # Bias
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 초기화
        nn.init.xavier_normal_(self.user_embedding_mf.weight)
        nn.init.xavier_normal_(self.item_embedding_mf.weight)
        nn.init.xavier_normal_(self.user_embedding_dnn.weight)
        nn.init.xavier_normal_(self.item_embedding_dnn.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, users, items, omega_explicit=0.5):
        """
        Forward pass
        users: [batch]
        items: [batch] or [batch, n_items]
        """
        # MF
        user_emb_mf = self.user_embedding_mf(users)
        item_emb_mf = self.item_embedding_mf(items)
        
        if item_emb_mf.dim() == 2:
            mf_output = (user_emb_mf * item_emb_mf).sum(dim=1)
        else:
            mf_output = torch.bmm(item_emb_mf, user_emb_mf.unsqueeze(2)).squeeze(2)
        
        # DNN
        user_emb_dnn = self.user_embedding_dnn(users)
        item_emb_dnn = self.item_embedding_dnn(items)
        
        if item_emb_dnn.dim() == 2:
            dnn_input = torch.cat([user_emb_dnn, item_emb_dnn], dim=1)
            dnn_output = self.dnn(dnn_input).squeeze(1)
        else:
            batch_size, n_items, _ = item_emb_dnn.shape
            user_expanded = user_emb_dnn.unsqueeze(1).expand(-1, n_items, -1)
            dnn_input = torch.cat([user_expanded, item_emb_dnn], dim=2)
            dnn_input = dnn_input.view(-1, self.embedding_dim * 2)
            dnn_output = self.dnn(dnn_input).view(batch_size, n_items)
        
        # Bias
        user_b = self.user_bias(users).squeeze(-1)
        
        if items.dim() == 1:
            item_b = self.item_bias(items).squeeze(-1)
        else:
            item_b = self.item_bias(items).squeeze(-1)
        
        bias = user_b.unsqueeze(1) if items.dim() > 1 else user_b
        bias = bias + item_b + self.global_bias
        
        # 융합
        omega_implicit = 1 - omega_explicit
        scores = omega_explicit * mf_output + omega_implicit * dnn_output + bias
        
        return scores


class SimpleTrainDataset(Dataset):
    """간단한 학습 데이터셋"""
    
    def __init__(self, train_df, num_items, num_negatives=4):
        self.train_df = train_df.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # User별 본 영화
        self.user_items_dict = train_df.groupby('user_idx')['movie_idx'].apply(
            lambda x: set(x.values)
        ).to_dict()
        
        self.all_items = np.arange(num_items, dtype=np.int64)
    
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        user = int(row['user_idx'])
        pos_item = int(row['movie_idx'])
        
        # ⚡⚡⚡ 최적화된 Negative sampling
        user_pos_items = self.user_items_dict.get(user, set())
        
        # 대부분 사용자는 적은 영화를 봤으므로 (평균 76개)
        # 전체에서 랜덤 샘플링 후 필터링이 더 빠름!
        if len(user_pos_items) < self.num_items * 0.3:  # 30% 미만
            # ⚡ 빠른 방법: 랜덤 샘플링 후 필터링
            neg_items = []
            attempts = 0
            while len(neg_items) < self.num_negatives and attempts < 10:
                candidates = np.random.randint(0, self.num_items, size=self.num_negatives * 2)
                valid = [c for c in candidates if c not in user_pos_items and c not in neg_items]
                neg_items.extend(valid[:self.num_negatives - len(neg_items)])
                attempts += 1
            
            # Fallback
            if len(neg_items) < self.num_negatives:
                neg_items.extend(np.random.choice(self.all_items, size=self.num_negatives - len(neg_items), replace=True))
            
            neg_items = np.array(neg_items[:self.num_negatives], dtype=np.int64)
        else:
            # 많이 본 사용자만 기존 방식 (느리지만 정확)
            neg_candidates = np.array([i for i in self.all_items if i not in user_pos_items])
            if len(neg_candidates) >= self.num_negatives:
                neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=False)
            else:
                neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=True)
        
        return user, pos_item, neg_items, 1.0, 1.0


def collate_train(batch):
    users, pos_items, neg_items_list, _, _ = zip(*batch)
    
    # ⚡ 최적화: numpy array 리스트를 먼저 numpy array로 변환
    neg_items_array = np.array(neg_items_list, dtype=np.int64)
    
    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.from_numpy(neg_items_array).long(),
        torch.FloatTensor([1.0] * len(users)),
        torch.FloatTensor([1.0] * len(users))
    )


class PaperEvaluationDataset(Dataset):
    """
    논문 방식 평가 데이터셋
    - 각 positive item당 99개 negative items 샘플링
    - 총 100개 아이템에서 랭킹
    """
    
    def __init__(self, test_df, train_df, num_items, num_negatives=99):
        self.test_df = test_df.reset_index(drop=True)
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Train에서 각 사용자가 본 아이템
        self.train_user_items = train_df.groupby('user_idx')['movie_idx'].apply(
            lambda x: set(x.values)
        ).to_dict()
        
        # 전체 아이템
        self.all_items = np.arange(num_items, dtype=np.int64)
    
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        """
        Returns:
            user: user index
            pos_item: positive item
            neg_items: 99 negative items
        """
        row = self.test_df.iloc[idx]
        user = int(row['user_idx'])
        pos_item = int(row['movie_idx'])
        
        # Negative sampling (train에 없는 아이템)
        train_items = self.train_user_items.get(user, set())
        train_items = train_items | {pos_item}  # positive도 제외
        
        # 후보 아이템
        neg_candidates = np.array([i for i in self.all_items if i not in train_items])
        
        if len(neg_candidates) >= self.num_negatives:
            neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=False)
        else:
            neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=True)
        
        return user, pos_item, neg_items


def collate_eval(batch):
    users, pos_items, neg_items_list = zip(*batch)
    
    # ⚡ 최적화: numpy array 리스트를 먼저 numpy array로 변환
    neg_items_array = np.array(neg_items_list, dtype=np.int64)
    
    return (
        torch.LongTensor(users),
        torch.LongTensor(pos_items),
        torch.from_numpy(neg_items_array).long()
    )


class MFDNN_Trainer:
    """논문 방식 MFDNN 학습"""
    
    def __init__(self, data_dir="data/processed", model_dir="data/models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # AMP Scaler
        if self.device.type == 'cuda' and AMP_AVAILABLE:
            if AMP_DEVICE is not None:
                self.scaler = GradScaler(AMP_DEVICE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.load_data()
    
    def load_data(self):
        """데이터 로드"""
        print("\n" + "=" * 50)
        print("Loading Data")
        print("=" * 50)
        
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.valid_df = pd.read_csv(self.data_dir / 'valid.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        with open(self.data_dir / 'mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        self.num_users = mappings['num_users']
        self.num_movies = mappings['num_movies']
        self.num_movies_train = mappings.get('num_movies_train', self.num_movies)
        
        print(f"✓ Users: {self.num_users:,}")
        print(f"✓ Movies (total): {self.num_movies:,}")
        if self.num_movies_train < self.num_movies:
            print(f"✓ Movies (train): {self.num_movies_train:,}")
            print(f"✓ Movies (eval-only): {self.num_movies - self.num_movies_train:,}")
        print(f"✓ Train: {len(self.train_df):,}")
        print(f"✓ Valid: {len(self.valid_df):,}")
        print(f"✓ Test: {len(self.test_df):,}")
    
    def train_epoch(self, model, dataloader, optimizer, omega_explicit, reg_weight):
        """1 epoch 학습 (BPR Loss)"""
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for users, pos_items, neg_items, _, _ in pbar:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward
            pos_scores = model(users, pos_items, omega_explicit)
            
            # Negative scores (batch)
            batch_size, num_neg = neg_items.shape
            users_expanded = users.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            neg_items_flat = neg_items.reshape(-1)
            neg_scores = model(users_expanded, neg_items_flat, omega_explicit)
            neg_scores = neg_scores.reshape(batch_size, num_neg)
            
            # BPR Loss
            pos_loss = -F.logsigmoid(pos_scores).mean()
            neg_loss = -F.logsigmoid(-neg_scores).mean()
            loss = pos_loss + neg_loss
            
            # Regularization
            if reg_weight > 0:
                reg_loss = reg_weight * (
                    model.user_embedding_mf.weight.pow(2).sum() +
                    model.item_embedding_mf.weight.pow(2).sum() +
                    model.user_embedding_dnn.weight.pow(2).sum() +
                    model.item_embedding_dnn.weight.pow(2).sum()
                ) / batch_size
                loss = loss + reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, model, eval_df, omega_explicit=0.5, k=10):
        """
        논문 방식 평가: 1 positive + 99 negatives
        """
        model.eval()
        
        # Evaluation dataset (학습용 영화만 사용)
        eval_dataset = PaperEvaluationDataset(eval_df, self.train_df, self.num_movies_train, num_negatives=99)
        eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False, collate_fn=collate_eval)
        
        ndcg_scores = []
        hr_scores = []
        
        for users, pos_items, neg_items in tqdm(eval_loader, desc="Evaluating"):
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            batch_size = users.size(0)
            
            # 1 positive + 99 negatives = 100 items
            all_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [batch, 100]
            
            # Score
            users_expanded = users.unsqueeze(1).expand(-1, 100).reshape(-1)
            items_flat = all_items.reshape(-1)
            scores = model(users_expanded, items_flat, omega_explicit)
            scores = scores.reshape(batch_size, 100).cpu().numpy()
            
            # Ranking (positive는 항상 index 0)
            for i in range(batch_size):
                # Rank
                ranks = np.argsort(-scores[i])  # 내림차순
                pos_rank = np.where(ranks == 0)[0][0]  # positive의 순위
                
                # HR@K
                if pos_rank < k:
                    hr_scores.append(1.0)
                    # NDCG@K
                    ndcg = 1.0 / np.log2(pos_rank + 2)
                    ndcg_scores.append(ndcg)
                else:
                    hr_scores.append(0.0)
                    ndcg_scores.append(0.0)
        
        return {
            'ndcg@10': np.mean(ndcg_scores),
            'hr@10': np.mean(hr_scores),
            'num_samples': len(ndcg_scores)
        }
    
    def train(self, embedding_dim=64, dnn_layers=[256, 128, 64],
              batch_size=256, lr=0.001, reg_weight=0.0001,
              omega_explicit=0.5, epochs=20, patience=5):
        """학습"""
        print("\n" + "=" * 50)
        print("Training MFDNN (Paper Style)")
        print("=" * 50)
        print(f"Embedding Dim: {embedding_dim}")
        print(f"DNN Layers: {dnn_layers}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {lr}")
        print(f"Epochs: {epochs}")
        
        # 모델 (학습용 영화만 사용)
        model = MFDNN(self.num_users, self.num_movies_train, embedding_dim, dnn_layers)
        model = model.to(self.device)
        
        print(f"✓ Model: {self.num_users:,} users × {self.num_movies_train:,} movies")
        
        # DataLoader (간단한 학습 데이터셋)
        train_dataset = SimpleTrainDataset(self.train_df, self.num_movies_train, num_negatives=4)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=2, collate_fn=collate_train, pin_memory=True,
                                 persistent_workers=False)  # ⚡ workers 4→2, persistent 제거
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # 학습
        history = []
        best_ndcg = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, omega_explicit, reg_weight)
            
            # Evaluate (Paper style)
            print("\nEvaluating (Paper Style: 1 pos + 99 neg)...")
            valid_metrics = self.evaluate(model, self.valid_df, omega_explicit, k=10)
            
            # History 저장
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'valid_ndcg': valid_metrics['ndcg@10'],
                'valid_hr': valid_metrics['hr@10']
            })
            
            scheduler.step(valid_metrics['ndcg@10'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid NDCG@10: {valid_metrics['ndcg@10']:.4f}, HR@10: {valid_metrics['hr@10']:.4f}")
            
            # Save best
            if valid_metrics['ndcg@10'] > best_ndcg:
                best_ndcg = valid_metrics['ndcg@10']
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'num_users': self.num_users,
                        'num_movies': self.num_movies,
                        'num_movies_train': self.num_movies_train,
                        'embedding_dim': embedding_dim,
                        'dnn_layers': dnn_layers,
                        'omega_explicit': omega_explicit
                    },
                    'valid_ndcg': best_ndcg,
                    'epoch': epoch,
                    'history': history
                }, self.model_dir / 'mfdnn_best.pt')
                
                print(f"✓ Best model saved (NDCG: {best_ndcg:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    print(f"   No improvement for {patience} consecutive epochs")
                    break
        
        # Save history
        with open(self.model_dir / 'history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        print(f"\n✓ Best Valid NDCG@10: {best_ndcg:.4f}")
        
        return model, history


def main():
    """메인 함수"""
    
    print("\n" + "=" * 70)
    print("MFDNN - Paper Exact Implementation")
    print("=" * 70)
    print("\n📄 Evaluation Method:")
    print("  - 1 positive + 99 random negatives = 100 items")
    print("  - Rank positive item among 100")
    print("  - Calculate NDCG@10, HR@10")
    
    trainer = MFDNN_Trainer(
        data_dir="data/processed",  # KMDB (Korean Movies)
        model_dir="data/models"
    )
    
    model, history = trainer.train(
        embedding_dim=64,
        dnn_layers=[256, 128, 64],
        batch_size=1024,
        lr=0.0001,
        reg_weight=0.0001,
        omega_explicit=0.5,
        epochs=50,
        patience=7
    )
    
    print("\n" + "=" * 70)
    print("✅ Training Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

