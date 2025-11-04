"""
BPR Matrix Factorization (간단하지만 강력한 baseline)
- LightGCN보다 단순하지만 효과적
- 많은 논문에서 강력한 baseline으로 사용
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
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

class BPR_MF(nn.Module):
    """
    Bayesian Personalized Ranking with Matrix Factorization
    - 가장 간단하지만 강력한 협업 필터링 모델
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and Item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # User and Item biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Xavier initialization
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, users, items):
        """
        Args:
            users: [batch_size]
            items: [batch_size] or [batch_size, num_items]
        
        Returns:
            scores: [batch_size] or [batch_size, num_items]
        """
        user_emb = self.user_embedding(users)  # [batch, dim]
        user_b = self.user_bias(users).squeeze()  # [batch]
        
        if items.dim() == 1:
            # Single item per user
            item_emb = self.item_embedding(items)  # [batch, dim]
            item_b = self.item_bias(items).squeeze()  # [batch]
            
            scores = (user_emb * item_emb).sum(dim=1) + user_b + item_b + self.global_bias
        else:
            # Multiple items per user
            item_emb = self.item_embedding(items)  # [batch, num_items, dim]
            item_b = self.item_bias(items).squeeze(-1)  # [batch, num_items]
            
            scores = torch.bmm(item_emb, user_emb.unsqueeze(2)).squeeze(2)  # [batch, num_items]
            scores = scores + user_b.unsqueeze(1) + item_b + self.global_bias
        
        return scores
    
    def predict(self, users, items):
        """Prediction for inference"""
        self.eval()
        with torch.no_grad():
            return self.forward(users, items)


class BPRDataset(Dataset):
    """BPR Loss를 위한 데이터셋"""
    
    def __init__(self, interactions, num_items, num_negatives=4):
        self.interactions = interactions
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # 각 유저가 본 영화들
        self.user_items = interactions.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        self.all_items = set(range(num_items))
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user = int(row['user_idx'])
        pos_item = int(row['movie_idx'])
        
        # Negative sampling
        user_pos_items = self.user_items.get(user, set())
        neg_candidates = list(self.all_items - user_pos_items)
        
        if len(neg_candidates) >= self.num_negatives:
            neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=False).tolist()
        else:
            neg_items = neg_candidates + [neg_candidates[0]] * (self.num_negatives - len(neg_candidates))
        
        return user, pos_item, neg_items


def collate_bpr(batch):
    users, pos_items, neg_items_list = zip(*batch)
    return (torch.LongTensor(users), 
            torch.LongTensor(pos_items), 
            torch.LongTensor(neg_items_list))


class BPR_MF_Trainer:
    """BPR-MF 학습 클래스"""
    
    def __init__(self, data_dir="data/processed", model_dir="data/models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.load_data()
    
    def load_data(self):
        """데이터 로드"""
        print("\n" + "=" * 50)
        print("Loading processed data...")
        print("=" * 50)
        
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.valid_df = pd.read_csv(self.data_dir / 'valid.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        with open(self.data_dir / 'mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        self.num_users = mappings['num_users']
        self.num_movies = mappings['num_movies']
        
        print(f"✓ Users: {self.num_users:,}")
        print(f"✓ Movies: {self.num_movies:,}")
        print(f"✓ Train: {len(self.train_df):,}")
        print(f"✓ Valid: {len(self.valid_df):,}")
        print(f"✓ Test: {len(self.test_df):,}")
    
    def get_bpr_loss(self, model, users, pos_items, neg_items, reg_weight=1e-5):
        """
        BPR Loss + L2 Regularization
        
        Args:
            users: [batch_size]
            pos_items: [batch_size]
            neg_items: [batch_size, num_negatives]
        """
        batch_size = users.size(0)
        num_negatives = neg_items.size(1)
        
        # Positive scores
        pos_scores = model(users, pos_items)  # [batch]
        
        # Negative scores
        users_expanded = users.unsqueeze(1).expand(-1, num_negatives)  # [batch, num_neg]
        neg_scores = model(users_expanded.contiguous().view(-1), 
                          neg_items.view(-1))  # [batch * num_neg]
        neg_scores = neg_scores.view(batch_size, num_negatives)  # [batch, num_neg]
        
        # BPR Loss: maximize pos_score - neg_score
        pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_negatives)
        bpr_loss = -F.logsigmoid(pos_scores_expanded - neg_scores).mean()
        
        # L2 Regularization
        user_emb = model.user_embedding(users)
        pos_item_emb = model.item_embedding(pos_items)
        neg_item_emb = model.item_embedding(neg_items.view(-1))
        
        reg_loss = reg_weight * (
            user_emb.pow(2).sum() + 
            pos_item_emb.pow(2).sum() + 
            neg_item_emb.pow(2).sum()
        ) / batch_size
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss, bpr_loss.item(), reg_loss.item()
    
    def train_epoch(self, model, dataloader, optimizer, reg_weight):
        """1 epoch 학습"""
        model.train()
        
        total_loss = 0
        total_bpr = 0
        total_reg = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for users, pos_items, neg_items in pbar:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            optimizer.zero_grad()
            
            loss, bpr_loss, reg_loss = self.get_bpr_loss(
                model, users, pos_items, neg_items, reg_weight
            )
            
            loss.backward()
            
            # Gradient clipping (안정적인 학습)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_bpr += bpr_loss
            total_reg += reg_loss
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpr': f'{bpr_loss:.4f}',
                'reg': f'{reg_loss:.6f}'
            })
        
        n = len(dataloader)
        return total_loss / n, total_bpr / n, total_reg / n
    
    @torch.no_grad()
    def evaluate(self, model, data_df, k=10, sample_users=None):
        """평가 (NDCG@K, Recall@K, Hit@K)"""
        model.eval()
        
        # User별 실제 정답 영화
        user_movies_true = data_df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        
        # 샘플링
        eval_users = list(user_movies_true.keys())
        if sample_users is not None and sample_users < len(eval_users):
            eval_users = np.random.choice(eval_users, size=sample_users, replace=False)
        
        # User별 이미 본 영화 (train set)
        train_user_movies = self.train_df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        
        ndcg_scores = []
        recall_scores = []
        hit_scores = []
        
        # 모든 영화 인덱스
        all_items = torch.arange(self.num_movies, device=self.device)
        
        for user_idx in tqdm(eval_users, desc="Evaluating", disable=True):
            true_items = user_movies_true.get(user_idx)
            if true_items is None or len(true_items) == 0:
                continue
            
            # 전체 영화에 대한 점수 계산
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            scores = model(user_tensor.expand(self.num_movies), all_items).cpu().numpy()
            
            # Train에서 이미 본 영화 제외
            train_items = train_user_movies.get(user_idx, set())
            scores[list(train_items)] = -np.inf
            
            # Top-K 추천
            top_k_items = np.argsort(scores)[::-1][:k]
            
            # Hit@K
            hit = 1.0 if len(set(top_k_items) & true_items) > 0 else 0.0
            hit_scores.append(hit)
            
            # Recall@K
            hits = len(set(top_k_items) & true_items)
            recall = hits / min(len(true_items), k)
            recall_scores.append(recall)
            
            # NDCG@K
            dcg = 0
            for i, item in enumerate(top_k_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return {
            'ndcg@10': np.mean(ndcg_scores),
            'recall@10': np.mean(recall_scores),
            'hit@10': np.mean(hit_scores),
            'num_users': len(ndcg_scores)
        }
    
    def train(self, embedding_dim=384, batch_size=32768, lr=0.01, 
              reg_weight=5e-5, epochs=20, patience=5, num_negatives=8, eval_sample_users=1000):
        """
        모델 학습
        
        하드웨어별 권장 설정:
        - CPU: embedding_dim=128, batch_size=8192, lr=0.001, num_negatives=8, epochs=20
        - GPU (RTX): embedding_dim=192, batch_size=16384, lr=0.001, num_negatives=12, epochs=25
        - A100: embedding_dim=256, batch_size=32768, lr=0.002, num_negatives=16, epochs=30 (기본값)
        - A100 (최대): embedding_dim=512, batch_size=65536, lr=0.003, num_negatives=24, epochs=50
        """
        print("\n" + "=" * 50)
        print("Training BPR-MF Model (A100 Optimized)")
        print("=" * 50)
        print(f"Embedding Dim: {embedding_dim}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {lr}")
        print(f"Reg Weight: {reg_weight}")
        print(f"Num Negatives: {num_negatives}")
        print(f"Epochs: {epochs}")
        print(f"Eval Sample Users: {eval_sample_users}")
        
        # GPU 확인
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  Running on CPU - Consider using smaller batch_size (8192) and embedding_dim (128)")
        
        # 모델 생성
        model = BPR_MF(self.num_users, self.num_movies, embedding_dim)
        model = model.to(self.device)
        
        # Dataset & DataLoader (A100 최적화)
        train_dataset = BPRDataset(self.train_df, self.num_movies, num_negatives=num_negatives)
        
        # GPU에 따라 num_workers 자동 조정
        if self.device.type == 'cuda':
            num_workers = 8  # A100: 8-16
        else:
            num_workers = 4  # CPU: 2-4
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_bpr, 
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"✓ DataLoader: batch_size={batch_size}, num_workers={num_workers}")
        
        # Optimizer (더 나은 최적화)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # 학습
        history = {
            'train_loss': [],
            'train_bpr': [],
            'train_reg': [],
            'valid_ndcg': [],
            'valid_recall': []
        }
        
        best_ndcg = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_bpr, train_reg = self.train_epoch(
                model, train_loader, optimizer, reg_weight
            )
            
            # Validate
            valid_metrics = self.evaluate(model, self.valid_df, k=10, sample_users=eval_sample_users)
            
            # History
            history['train_loss'].append(train_loss)
            history['train_bpr'].append(train_bpr)
            history['train_reg'].append(train_reg)
            history['valid_ndcg'].append(valid_metrics['ndcg@10'])
            history['valid_recall'].append(valid_metrics['recall@10'])
            
            # Scheduler
            scheduler.step(valid_metrics['ndcg@10'])
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} (BPR: {train_bpr:.4f}, Reg: {train_reg:.6f})")
            print(f"Valid NDCG@10: {valid_metrics['ndcg@10']:.4f}, Recall@10: {valid_metrics['recall@10']:.4f}, Hit@10: {valid_metrics['hit@10']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if valid_metrics['ndcg@10'] > best_ndcg:
                best_ndcg = valid_metrics['ndcg@10']
                patience_counter = 0
                
                # 모델 저장
                save_path = self.model_dir / 'bprmf_best.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'num_users': self.num_users,
                        'num_movies': self.num_movies,
                        'embedding_dim': embedding_dim
                    },
                    'valid_ndcg': best_ndcg,
                    'epoch': epoch
                }, save_path)
                print(f"✓ Best model saved (NDCG: {best_ndcg:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Save history
        with open(self.model_dir / 'bprmf_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        # Plot
        self.plot_history(history)
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best Valid NDCG@10: {best_ndcg:.4f}")
        print("=" * 50)
        
        return model, history
    
    def plot_history(self, history):
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Total Loss', marker='o')
        axes[0].plot(history['train_bpr'], label='BPR Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('BPR-MF - Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics
        axes[1].plot(history['valid_ndcg'], label='NDCG@10', marker='o')
        axes[1].plot(history['valid_recall'], label='Recall@10', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('BPR-MF - Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'bprmf_training_history.png', dpi=150)
        print(f"✓ Training history plot saved")


def main():
    """메인 함수"""
    
    trainer = BPR_MF_Trainer()
    
    # BPR-MF 학습
    print("\n" + "=" * 70)
    print("TRAINING BPR Matrix Factorization")
    print("=" * 70)
    
    # A100 GPU 최적화 설정 (최종 시도 - 더 강력한 모델)
    # 더 큰 모델, 더 높은 LR, 더 오래 학습
    model, history = trainer.train(
        embedding_dim=256,      # 64 → 256 (더 큰 모델)
        batch_size=16384,       # 16384 유지
        lr=0.001,              # 0.0005 → 0.001 (2배 증가)
        reg_weight=5e-5,       # 1e-4 → 5e-5 (정규화 완화)
        num_negatives=8,       # 8 유지
        epochs=50,             # 30 → 50 (더 오래 학습)
        patience=7,            # 3 → 7 (더 오래 기다림)
        eval_sample_users=None  # 전체 사용자 평가  
    )
    
    # CPU/일반 GPU용 설정 (주석 해제하여 사용)
    # model, history = trainer.train(
    #     embedding_dim=128,
    #     batch_size=8192,
    #     lr=0.001,
    #     num_negatives=8,
    #     epochs=20,
    #     patience=5,
    #     eval_sample_users=500
    # )
    
    # Test set 평가
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    checkpoint = torch.load(trainer.model_dir / 'bprmf_best.pt',
                           map_location=trainer.device,
                           weights_only=False)
    
    test_model = BPR_MF(
        checkpoint['config']['num_users'],
        checkpoint['config']['num_movies'],
        checkpoint['config']['embedding_dim']
    ).to(trainer.device)
    
    test_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n[1/2] NDCG@10 (Top 10 quality)...")
    test_metrics_10 = trainer.evaluate(test_model, trainer.test_df, k=10, sample_users=None)
    
    print("\n[2/2] NDCG@50 (Top 50 quality)...")
    test_metrics_50 = trainer.evaluate(test_model, trainer.test_df, k=50, sample_users=None)
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (BPR-MF)")
    print("=" * 70)
    print(f"\n📊 Top-10 Metrics:")
    print(f"   NDCG@10:   {test_metrics_10['ndcg@10']:.4f}  {'✅ PASS' if test_metrics_10['ndcg@10'] >= 0.4 else '⚠️  NEEDS IMPROVEMENT'}")
    print(f"   Recall@10: {test_metrics_10['recall@10']:.4f}")
    print(f"   Hit@10:    {test_metrics_10['hit@10']:.4f}")
    print(f"   Users evaluated: {test_metrics_10['num_users']}")
    
    print(f"\n📊 Top-50 Metrics:")
    print(f"   NDCG@50:   {test_metrics_50['ndcg@10']:.4f}")
    print(f"   Recall@50: {test_metrics_50['recall@10']:.4f}")
    print(f"   Hit@50:    {test_metrics_50['hit@10']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

