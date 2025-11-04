"""
BERT4Rec (Fixed Version)
- Fixed index mismatch bug
- Simplified item indexing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import math
import time

print("=" * 70)
print("BERT4Rec (Fixed Version)")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


# ============================================================
# Model Components (Same as before)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class BERT4Rec(nn.Module):
    """
    BERT4Rec Model (Fixed)
    
    Item indexing (SIMPLIFIED):
    - 0: PAD token
    - 1~num_items: Real items
    - num_items+1: MASK token
    """
    
    def __init__(self, num_items, max_seq_len=50, d_model=64, num_heads=2, 
                 num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Tokens: 0=PAD, 1~num_items=items, num_items+1=MASK
        self.item_embedding = nn.Embedding(num_items + 2, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output: predict items 0~num_items-1 (original indices)
        self.out = nn.Linear(d_model, num_items)
        self.dropout = nn.Dropout(dropout)
        
        self.pad_token = 0
        self.mask_token = num_items + 1
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len] - token ids (0=PAD, 1~num_items=items, num_items+1=MASK)
        Returns:
            logits: [batch, seq_len, num_items] - predictions for items 0~num_items-1
        """
        x = self.item_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        logits = self.out(x)
        return logits


# ============================================================
# Dataset (FIXED!)
# ============================================================

class BERT4RecDataset(Dataset):
    """
    BERT4Rec Dataset with Masking (Fixed)
    
    Item mapping:
    - Original items: 0~num_items-1
    - Model tokens: 1~num_items (shifted by +1)
    - PAD: 0
    - MASK: num_items+1
    """
    
    def __init__(self, sequences, num_items, max_seq_len=50, mask_prob=0.15):
        self.sequences = sequences
        self.user_ids = list(sequences.keys())
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        self.pad_token = 0
        self.mask_token = num_items + 1
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id].copy()
        
        # Truncate
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        
        seq_len = len(seq)
        
        # Shift items: 0~num_items-1 → 1~num_items
        seq_shifted = [x + 1 for x in seq]
        
        # Masking
        masked_seq = seq_shifted.copy()
        labels = []  # Only for actual items
        
        for i in range(seq_len):
            if np.random.random() < self.mask_prob:
                labels.append(seq[i])  # Original label (0~num_items-1)
                
                prob = np.random.random()
                if prob < 0.8:
                    masked_seq[i] = self.mask_token
                elif prob < 0.9:
                    masked_seq[i] = np.random.randint(1, self.num_items + 1)
            else:
                labels.append(-1)  # Not masked (use -1 instead of 0)
        
        # Padding (앞에 추가)
        if len(masked_seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(masked_seq)
            masked_seq = [self.pad_token] * pad_len + masked_seq
            labels = [-1] * pad_len + labels  # -1 for padding
        
        return {
            'input': np.array(masked_seq, dtype=np.long),
            'label': np.array(labels, dtype=np.long),
            'user_id': user_id
        }


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        
        mask = (inputs != model.pad_token).long()
        
        optimizer.zero_grad()
        logits = model(inputs, mask)
        
        # Loss: cross-entropy on masked positions
        # Labels: -1=ignore, 0~num_items-1=target
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-1
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accuracy (on masked positions only)
        predictions = logits.argmax(dim=-1)  # 0~num_items-1
        mask_positions = (labels >= 0)
        correct = ((predictions == labels) & mask_positions).sum().item()
        count = mask_positions.sum().item()
        
        total_correct += correct
        total_count += count
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate(model, test_sequences, train_sequences, num_items, device, k=10):
    model.eval()
    
    ndcg_scores = []
    recall_scores = []
    hit_scores = []
    
    for user_idx, target_item in tqdm(test_sequences.items(), desc="Evaluating"):
        train_seq = train_sequences.get(user_idx, [])
        if len(train_seq) == 0:
            continue
        
        # Prepare sequence (shift by +1)
        seq = [x + 1 for x in train_seq[-model.max_seq_len:]]
        
        # Pad
        if len(seq) < model.max_seq_len:
            seq = [model.pad_token] * (model.max_seq_len - len(seq)) + seq
        
        # Predict
        x = torch.LongTensor([seq]).to(device)
        mask = (x != model.pad_token).long()
        logits = model(x, mask)
        
        # Get last position scores
        scores = logits[0, -1, :].cpu().numpy()  # [num_items]
        
        # Mask training items (convert to 0-indexed)
        for item in train_seq:
            if 0 <= item < num_items:
                scores[item] = -np.inf
        
        # Top-K (scores are 0-indexed)
        top_k = np.argsort(scores)[::-1][:k]
        
        # Compare with target (target_item is 0-indexed)
        hit = 1.0 if target_item in top_k else 0.0
        hit_scores.append(hit)
        recall_scores.append(hit)
        
        if target_item in top_k:
            rank = np.where(top_k == target_item)[0][0]
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        ndcg_scores.append(ndcg)
    
    return {
        'ndcg@10': np.mean(ndcg_scores),
        'recall@10': np.mean(recall_scores),
        'hit@10': np.mean(hit_scores),
        'num_users': len(ndcg_scores)
    }


# ============================================================
# Main
# ============================================================

def main():
    data_dir = Path("data/processed_sequential")
    model_dir = Path("data/models")
    model_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Loading Data...")
    print("=" * 70)
    
    with open(data_dir / 'sequences.pkl', 'rb') as f:
        seq_data = pickle.load(f)
    
    with open(data_dir / 'mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    train_sequences = seq_data['train_sequences']
    valid_sequences = seq_data['valid_sequences']
    test_sequences = seq_data['test_sequences']
    max_seq_len = seq_data['max_seq_length']
    num_items = mappings['num_movies']
    
    print(f"✓ Users: {len(train_sequences):,}")
    print(f"✓ Items: {num_items:,}")
    print(f"✓ Max seq len: {max_seq_len}")
    
    # Config
    config = {
        'd_model': 128,        # Increased
        'num_heads': 4,        # Increased
        'num_layers': 3,       # Increased
        'd_ff': 512,          # Increased
        'dropout': 0.2,
        'mask_prob': 0.2,     # Increased
        'batch_size': 512,    # Increased
        'lr': 0.001,
        'epochs': 100,
        'patience': 10,
    }
    
    print("\n" + "=" * 70)
    print("Hyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    
    # Model
    model = BERT4Rec(
        num_items=num_items,
        max_seq_len=max_seq_len,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model parameters: {num_params:,}")
    
    # Dataset
    train_dataset = BERT4RecDataset(
        train_sequences, num_items, max_seq_len, config['mask_prob']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
    )
    
    # Training
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    best_ndcg = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        valid_metrics = evaluate(model, valid_sequences, train_sequences, num_items, device)
        
        scheduler.step(valid_metrics['ndcg@10'])
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTrain Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Valid NDCG@10: {valid_metrics['ndcg@10']:.4f}")
        print(f"Valid Recall@10: {valid_metrics['recall@10']:.4f}")
        print(f"LR: {lr:.6f}")
        
        if valid_metrics['ndcg@10'] > best_ndcg:
            best_ndcg = valid_metrics['ndcg@10']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'ndcg': best_ndcg
            }, model_dir / 'bert4rec_fixed_best.pt')
            print(f"✓ Best model saved (NDCG: {best_ndcg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping!")
                break
    
    # Test
    print("\n" + "=" * 70)
    print("Test Evaluation")
    print("=" * 70)
    
    checkpoint = torch.load(model_dir / 'bert4rec_fixed_best.pt', 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_sequences, train_sequences, num_items, device)
    
    print(f"\n📊 Final Test Results:")
    print(f"   NDCG@10:   {test_metrics['ndcg@10']:.4f}")
    print(f"   Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"   Hit@10:    {test_metrics['hit@10']:.4f}")
    
    if test_metrics['ndcg@10'] >= 0.15:
        print("\n✅ Success! NDCG >= 0.15")
    elif test_metrics['ndcg@10'] >= 0.10:
        print("\n⚠️  Acceptable. NDCG >= 0.10")
    else:
        print("\n❌ Low performance.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

