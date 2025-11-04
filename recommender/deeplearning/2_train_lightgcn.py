"""
LightGCN 기반 추천 시스템
- Graph Neural Network 기반 협업 필터링
- He et al., SIGIR 2020
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
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    from torch_geometric.nn.conv import LGConv
    USE_PYG = True
except ImportError:
    USE_PYG = False
    print("⚠️ PyTorch Geometric not found. Using manual implementation.")


# ============================================================
# LightGCN Model
# ============================================================

class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network
    - No feature transformation
    - No non-linear activation
    - Only neighborhood aggregation
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Learnable embeddings (초기값만 학습, propagation은 고정)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # LGConv layer (PyG 사용 가능하면)
        if USE_PYG:
            self.convs = nn.ModuleList([
                LGConv(normalize=True) for _ in range(num_layers)
            ])
    
    def forward(self, edge_index):
        """
        Graph Convolution을 통한 embedding 전파
        
        Args:
            edge_index: [2, num_edges] COO format
        Returns:
            user_emb_final: [num_users, embedding_dim]
            item_emb_final: [num_items, embedding_dim]
        """
        # 초기 embeddings
        user_emb_0 = self.user_embedding.weight
        item_emb_0 = self.item_embedding.weight
        all_emb_0 = torch.cat([user_emb_0, item_emb_0], dim=0)
        
        # Layer-wise propagation
        all_embs = [all_emb_0]
        current_emb = all_emb_0
        
        if USE_PYG:
            # PyTorch Geometric 사용
            for conv in self.convs:
                current_emb = conv(current_emb, edge_index)
                all_embs.append(current_emb)
        else:
            # Manual implementation with sparse matrix multiplication
            # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
            num_nodes = self.num_users + self.num_items
            
            # Calculate degrees
            degrees = torch.zeros(num_nodes, device=edge_index.device)
            degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
            degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
            
            # D^(-1/2)
            deg_inv_sqrt = degrees.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # Normalize edge weights: D^(-1/2)_src * D^(-1/2)_dst
            edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
            
            # Create sparse adjacency matrix
            num_edges = edge_index.size(1)
            adj_matrix = torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                (num_nodes, num_nodes),
                device=edge_index.device
            )
            
            # Propagation using sparse matrix multiplication
            for _ in range(self.num_layers):
                current_emb = torch.sparse.mm(adj_matrix, current_emb)
                all_embs.append(current_emb)
        
        # Mean pooling across all layers
        all_emb_final = torch.mean(torch.stack(all_embs, dim=0), dim=0)
        
        # Split back to user and item embeddings
        user_emb_final = all_emb_final[:self.num_users]
        item_emb_final = all_emb_final[self.num_users:]
        
        return user_emb_final, item_emb_final, user_emb_0, item_emb_0
    
    def get_scores(self, user_emb, item_emb):
        """User-Item 점수 계산"""
        return (user_emb * item_emb).sum(dim=1)


# ============================================================
# Dataset
# ============================================================

class BPRDataset(Dataset):
    """BPR Loss를 위한 데이터셋"""
    
    def __init__(self, interactions, num_items, num_negatives=8):
        self.interactions = interactions
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # 각 유저가 본 영화들
        self.user_items = interactions.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        
        # 모든 영화 ID set (negative sampling 최적화)
        self.all_items = set(range(num_items))
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user = int(row['user_idx'])
        pos_item = int(row['movie_idx'])
        
        # Negative sampling (더 효율적으로 개선)
        user_pos_items = self.user_items.get(user, set())
        neg_candidates = list(self.all_items - user_pos_items)
        
        # 랜덤하게 negative samples 선택
        if len(neg_candidates) >= self.num_negatives:
            neg_items = np.random.choice(neg_candidates, size=self.num_negatives, replace=False).tolist()
        else:
            # 충분한 negative sample이 없는 경우 (매우 드물음)
            neg_items = neg_candidates + [neg_candidates[0]] * (self.num_negatives - len(neg_candidates))
        
        return user, pos_item, neg_items


def collate_bpr(batch):
    users, pos_items, neg_items_list = zip(*batch)
    return (torch.LongTensor(users), 
            torch.LongTensor(pos_items), 
            torch.LongTensor(neg_items_list))


# ============================================================
# Trainer
# ============================================================

class LightGCNTrainer:
    """LightGCN 학습 클래스"""
    
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
    
    def create_graph(self):
        """User-Item bipartite graph 생성"""
        users = self.train_df['user_idx'].values
        items = self.train_df['movie_idx'].values + self.num_users
        
        # Bidirectional edges
        edge_index = np.array([
            np.concatenate([users, items]),
            np.concatenate([items, users])
        ])
        
        return torch.from_numpy(edge_index).long()
    
    def get_bpr_loss(self, model, users, pos_items, neg_items, user_emb, item_emb, 
                     init_user_emb, init_item_emb, reg_weight=1e-4):
        """
        BPR Loss + L2 Regularization
        """
        batch_size = users.size(0)
        num_negatives = neg_items.size(1)
        
        # Positive scores
        pos_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        pos_scores = model.get_scores(pos_user_emb, pos_item_emb)
        
        # Negative scores
        neg_item_emb = item_emb[neg_items]  # [batch, num_neg, dim]
        neg_scores = torch.bmm(neg_item_emb, pos_user_emb.unsqueeze(2)).squeeze(2)
        
        # BPR Loss
        pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_negatives)
        bpr_loss = -F.logsigmoid(pos_scores_expanded - neg_scores).mean()
        
        # L2 Regularization (초기 embedding에만 적용)
        reg_loss = reg_weight * (
            init_user_emb[users].pow(2).sum() + 
            init_item_emb[pos_items].pow(2).sum() +
            init_item_emb[neg_items.view(-1)].pow(2).sum()
        ) / batch_size
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss, bpr_loss.item(), reg_loss.item()
    
    def train_epoch(self, model, edge_index, dataloader, optimizer, reg_weight):
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
            
            # Graph convolution (매 배치마다 수행 - gradient 흐름 유지)
            user_emb, item_emb, init_user_emb, init_item_emb = model(edge_index)
            
            loss, bpr_loss, reg_loss = self.get_bpr_loss(
                model, users, pos_items, neg_items,
                user_emb, item_emb, init_user_emb, init_item_emb,
                reg_weight
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
    def evaluate(self, model, edge_index, data_df, k=10, verbose=False, sample_users=None):
        """
        평가 (NDCG@K, Recall@K, Hit@K)
        
        Args:
            sample_users: 평가할 사용자 수 (None이면 전체, 숫자면 샘플링)
        """
        model.eval()
        
        # Graph convolution
        user_emb, item_emb, _, _ = model(edge_index)
        
        # User별 실제 정답 영화
        user_movies_true = data_df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        
        # 샘플링 (학습 중에는 빠른 평가를 위해)
        eval_users = list(user_movies_true.keys())
        if sample_users is not None and sample_users < len(eval_users):
            eval_users = np.random.choice(eval_users, size=sample_users, replace=False)
        
        # User별 이미 본 영화 (train set)
        train_user_movies = self.train_df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
        
        ndcg_scores = []
        recall_scores = []
        hit_scores = []
        
        for user_idx in tqdm(eval_users, desc="Evaluating", disable=not verbose):
            true_items = user_movies_true.get(user_idx)
            if true_items is None or len(true_items) == 0:
                continue
            
            # 전체 영화에 대한 점수 계산
            u_emb = user_emb[user_idx].unsqueeze(0)
            scores = (u_emb * item_emb).sum(dim=1).cpu().numpy()
            
            # Train에서 이미 본 영화 제외
            train_items = train_user_movies.get(user_idx, set())
            scores[list(train_items)] = -np.inf
            
            # Top-K 추천
            top_k_items = np.argsort(scores)[::-1][:k]
            
            # Hit@K (적어도 하나라도 맞췄는지)
            hit = 1.0 if len(set(top_k_items) & true_items) > 0 else 0.0
            hit_scores.append(hit)
            
            # Recall@K
            hits = len(set(top_k_items) & true_items)
            recall = hits / min(len(true_items), k)
            recall_scores.append(recall)
            
            # NDCG@K (정확한 계산)
            dcg = 0
            for i, item in enumerate(top_k_items):
                if item in true_items:
                    # Relevance = 1 for binary feedback
                    dcg += 1.0 / np.log2(i + 2)
            
            # Ideal DCG (모든 관련 아이템이 상위에 있을 때)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return {
            'ndcg@10': np.mean(ndcg_scores),
            'recall@10': np.mean(recall_scores),
            'hit@10': np.mean(hit_scores),
            'num_users': len(ndcg_scores)
        }
    
    def train(self, embedding_dim=384, num_layers=3, batch_size=32768, 
              lr=0.01, reg_weight=5e-5, epochs=20, patience=5, num_negatives=8,
              eval_sample_users=1000):
        """
        모델 학습 (A100 최적화)
        
        하드웨어별 권장 설정:
        - CPU: embedding_dim=128, batch_size=8192, lr=0.001, num_negatives=8, epochs=20
        - GPU (RTX): embedding_dim=192, batch_size=16384, lr=0.001, num_negatives=12, epochs=25
        - A100: embedding_dim=256, batch_size=32768, lr=0.002, num_negatives=16, epochs=30 (기본값)
        - A100 (최대): embedding_dim=512, batch_size=65536, lr=0.003, num_negatives=24, epochs=50
        
        Args:
            eval_sample_users: 평가 시 샘플링할 사용자 수 (None이면 전체)
        """
        print("\n" + "=" * 50)
        print("Training LightGCN Model (A100 Optimized)")
        print("=" * 50)
        print(f"Embedding Dim: {embedding_dim}")
        print(f"Num Layers: {num_layers}")
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
        model = LightGCN(self.num_users, self.num_movies, embedding_dim, num_layers)
        model = model.to(self.device)
        
        # Graph 생성
        print("\nCreating graph...")
        edge_index = self.create_graph().to(self.device)
        print(f"✓ Graph created: {edge_index.size(1):,} edges")
        
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        
        # Scheduler (더 점진적으로 학습률 감소)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
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
                model, edge_index, train_loader, optimizer, reg_weight
            )
            
            # Validate (샘플링으로 빠른 평가)
            valid_metrics = self.evaluate(
                model, edge_index, self.valid_df, k=10, 
                verbose=False, sample_users=eval_sample_users
            )
            
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
                save_path = self.model_dir / 'lightgcn_best.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'edge_index': edge_index.cpu(),
                    'config': {
                        'num_users': self.num_users,
                        'num_movies': self.num_movies,
                        'embedding_dim': embedding_dim,
                        'num_layers': num_layers
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
        with open(self.model_dir / 'lightgcn_history.pkl', 'wb') as f:
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
        axes[0].set_title('LightGCN - Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics
        axes[1].plot(history['valid_ndcg'], label='NDCG@10', marker='o')
        axes[1].plot(history['valid_recall'], label='Recall@10', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('LightGCN - Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'lightgcn_training_history.png', dpi=150)
        print(f"✓ Training history plot saved")


def main():
    """메인 함수"""
    
    trainer = LightGCNTrainer()
    
    # LightGCN 학습
    print("\n" + "=" * 70)
    print("TRAINING LightGCN (Graph Neural Network)")
    print("=" * 70)
    
    # A100 GPU 최적화 설정 (수정됨 - Overfitting 방지)
    # 낮은 LR로 안정적 학습 (NDCG 0.20-0.35 목표)
    model, history = trainer.train(
        embedding_dim=128,      # 384 → 128 (Overfitting 방지)
        num_layers=2,           # 3 → 2 (더 작은 모델)
        batch_size=16384,       # 32768 → 16384 (안정성)
        lr=0.0005,             # 0.01 → 0.0005 ⭐ 가장 중요! (20배 감소)
        reg_weight=1e-4,       # 5e-5 → 1e-4 (더 강한 정규화)
        num_negatives=8,       # 8 유지
        epochs=30,             # 충분한 학습
        patience=3,            # 5 → 3 (더 빠른 early stopping)
        eval_sample_users=None  # 1000 → None (전체 사용자 평가로 일관성)  
    )
    
    # CPU/일반 GPU용 설정 (주석 해제하여 사용)
    # model, history = trainer.train(
    #     embedding_dim=128,
    #     num_layers=3,
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
    
    checkpoint = torch.load(trainer.model_dir / 'lightgcn_best.pt',
                           map_location=trainer.device, weights_only=False)
    
    test_model = LightGCN(
        checkpoint['config']['num_users'],
        checkpoint['config']['num_movies'],
        checkpoint['config']['embedding_dim'],
        checkpoint['config']['num_layers']
    ).to(trainer.device)
    
    test_model.load_state_dict(checkpoint['model_state_dict'])
    edge_index = checkpoint['edge_index'].to(trainer.device)
    
    print("\n" + "=" * 70)
    print("Evaluating on TEST set (FULL evaluation)...")
    print("=" * 70)
    
    print("\n[1/2] NDCG@10 (Top 10 quality)...")
    test_metrics_10 = trainer.evaluate(test_model, edge_index, trainer.test_df, k=10, verbose=True, sample_users=None)
    
    print("\n[2/2] NDCG@50 (Top 50 quality for LLM filtering)...")
    test_metrics_50 = trainer.evaluate(test_model, edge_index, trainer.test_df, k=50, verbose=True, sample_users=None)
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"\n📊 Top-10 Metrics (User's actual view):")
    print(f"   NDCG@10:   {test_metrics_10['ndcg@10']:.4f}  {'✅ PASS' if test_metrics_10['ndcg@10'] >= 0.4 else '⚠️  NEEDS IMPROVEMENT'}")
    print(f"   Recall@10: {test_metrics_10['recall@10']:.4f}")
    print(f"   Hit@10:    {test_metrics_10['hit@10']:.4f}")
    print(f"   Users evaluated: {test_metrics_10['num_users']}")
    
    print(f"\n📊 Top-50 Metrics (For LLM filtering):")
    print(f"   NDCG@50:   {test_metrics_50['ndcg@10']:.4f}")
    print(f"   Recall@50: {test_metrics_50['recall@10']:.4f}")
    print(f"   Hit@50:    {test_metrics_50['hit@10']:.4f}")
    
    print("\n" + "=" * 70)
    print("💡 Interpretation:")
    print("  - NDCG@10 ≥ 0.4: SUCCESS! ✅")
    print("  - NDCG@10 < 0.4: Need further tuning ⚠️")
    print("=" * 70)


if __name__ == "__main__":
    main()
