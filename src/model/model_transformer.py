import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
import math
from utilities.logging_config import setup_logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

setup_logging()
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes, embed_dim):
        super(EmbeddingLayer, self).__init__()
        
        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        
        self.embeddings = nn.ModuleDict()
        for col_name, vocab_size in vocab_sizes.items():
            self.embeddings[col_name] = nn.Embedding(vocab_size, embed_dim)
        
        logger.info(f"Embedding layers creati:")
        for col_name, vocab_size in vocab_sizes.items():
            logger.info(f"  {col_name}: {vocab_size} -> {embed_dim}")
    
    def forward(self, categorical_inputs):
        embedded_features = []
        
        for col_name, col_tensor in categorical_inputs.items():
            if col_name in self.embeddings:
                embedded = self.embeddings[col_name](col_tensor)
                embedded_features.append(embedded)
        
        if embedded_features:
            return torch.cat(embedded_features, dim=-1)
        else:
            batch_size, seq_len = next(iter(categorical_inputs.values())).shape
            return torch.zeros(batch_size, seq_len, 0, device=next(iter(categorical_inputs.values())).device)


class NetworkTrafficTransformer(nn.Module):
    def __init__(self, config, n_classes, feature_groups, vocab_sizes=None):
        super(NetworkTrafficTransformer, self).__init__()
        
        self.config = config
        self.n_classes = n_classes
        self.feature_groups = feature_groups
        self.vocab_sizes = vocab_sizes or {}
        
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.embed_dim = config.get('categorical_embed_dim', 16)
        
        self.num_numeric = len(feature_groups['numeric']['columns'])
        self.num_categorical = len(feature_groups['categorical']['columns'])
        
        if self.num_categorical > 0 and self.vocab_sizes:
            self.categorical_embedding = EmbeddingLayer(self.vocab_sizes, self.embed_dim)
            self.categorical_dim = self.num_categorical * self.embed_dim
        else:
            self.categorical_embedding = None
            self.categorical_dim = 0
        
        self.input_dim = self.num_numeric + self.categorical_dim
        
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=512)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, n_classes)
        )
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Transformer creato:")
        logger.info(f"  Parametri totali: {total_params:,}")
        logger.info(f"  Classi: {n_classes}")
        logger.info(f"  d_model: {self.d_model}")
        logger.info(f"  Layers: {self.num_layers}")
        logger.info(f"  Heads: {self.nhead}")
        logger.info(f"  Input dim: {self.input_dim} (num: {self.num_numeric}, cat: {self.categorical_dim})")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, numeric_features, categorical_features=None):
        batch_size, seq_len, _ = numeric_features.shape
        
        if self.categorical_embedding is not None and categorical_features is not None:
            categorical_embedded = self.categorical_embedding(categorical_features)
            combined_features = torch.cat([numeric_features, categorical_embedded], dim=-1)
        else:
            combined_features = numeric_features
        
        x = self.input_projection(combined_features)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = self.layer_norm(x)
        
        last_hidden = x[:, -1, :]
        
        output = self.classifier(last_hidden)
        
        return output

class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window_size, seed):
        self.X = X
        self.y = y
        self.window_size = window_size
        
        self.window_starts = list(range(len(X) - window_size + 1))
        
        import random
        random.seed(seed)
        random.shuffle(self.window_starts)
        
    def __len__(self):
        return len(self.window_starts)
    
    def __getitem__(self, idx):
        start_idx = self.window_starts[idx]
        end_idx = start_idx + self.window_size
        
        # Crea la seqKuenza
        sequence = torch.FloatTensor(self.X[start_idx:end_idx])
        target = torch.LongTensor([self.y[end_idx - 1]])[0]
        
        return sequence, target
    
    def reshuffle(self, seed):
        import random
        random.seed(seed)
        random.shuffle(self.window_starts)


class NetworkTrafficDatasetTransformer:
    
    def __init__(self, config_path="config/dataset.json", hyperparams_path="config/hyperparameters.json", 
                 model_size="small", metadata_path="resources/datasets/transformer_metadata.json"):
        self.metadata_path = metadata_path
        self.load_configs(config_path, hyperparams_path, model_size)
        
    def load_configs(self, config_path, hyperparams_path, model_size="small"):

        try:
            with open(config_path, 'r') as f:
                self.dataset_config = json.load(f)['dataset']
        except:
            logger.warning(f"Impossibile caricare {config_path}, usando config di default")
            self.dataset_config = {}
        
        try:
            with open(hyperparams_path, 'r') as f:
                hyperparams_data = json.load(f)
                if 'transformer' in hyperparams_data:
                    self.hyperparams = hyperparams_data['transformer'][model_size]
                else:
                    self.hyperparams = {
                        'batch_size': 32,
                        'lr': 0.0003,
                        'weight_decay': 1e-4,
                        'd_model': 128,
                        'nhead': 8,
                        'num_layers': 4,
                        'dropout': 0.1,
                        'categorical_embed_dim': 16
                    }
        except:
            logger.warning(f"Impossibile caricare {hyperparams_path}, usando hyperparams transformer di default")
            self.hyperparams = {
                'batch_size': 32,
                'lr': 0.0003,
                'weight_decay': 1e-4,
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'categorical_embed_dim': 16
            }
        
        with open(self.metadata_path, 'r') as f:
            self.temporal_metadata = json.load(f)
            
        self.feature_columns = self.temporal_metadata['feature_columns']
        self.n_classes = self.temporal_metadata['n_classes']
        self.class_mapping = self.temporal_metadata['class_mapping']
        self.feature_groups = self.temporal_metadata.get('feature_groups', {})
        self.sequence_length = self.temporal_metadata['temporal_config']['sequence_length']
        
        if 'embedding_config' in self.temporal_metadata:
            vocab_stats = self.temporal_metadata['embedding_config'].get('vocab_stats', {})
            self.vocab_sizes = {col: stats['vocab_size'] for col, stats in vocab_stats.items()}
        else:
            self.vocab_sizes = {}
        
        logger.info(f"Configurazione Transformer caricata:")
        logger.info(f"  Feature columns: {len(self.feature_columns)}")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Classi: {self.n_classes}")
        logger.info(f"  Vocab sizes: {self.vocab_sizes}")
        logger.info(f"  Hyperparams: {self.hyperparams}")
    
    def load_processed_data(self, train_csv_path, val_csv_path, test_csv_path):
       
        logger.info("Caricamento dataset processati...")
        
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)
        
        logger.info(f"Dataset processati caricati:")
        logger.info(f"  Training: {train_df.shape[0]:,} campioni")
        logger.info(f"  Validation: {val_df.shape[0]:,} campioni") 
        logger.info(f"  Test: {test_df.shape[0]:,} campioni")
        
        self.X_train = train_df[self.feature_columns].values.astype(np.float32)
        self.y_train = train_df['multiclass_target'].values.astype(np.int64)
        
        self.X_val = val_df[self.feature_columns].values.astype(np.float32)
        self.y_val = val_df['multiclass_target'].values.astype(np.int64)
        
        self.X_test = test_df[self.feature_columns].values.astype(np.float32)
        self.y_test = test_df['multiclass_target'].values.astype(np.int64)
        
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Class weights calcolati: {class_weights}")
        except Exception as e:
            logger.warning(f"Impossibile calcolare class weights: {e}")
            self.class_weights = None
        
        return self.X_train.shape[1]

    def separate_features(self, sequences):
        numeric_cols = self.feature_groups.get('numeric', {}).get('columns', [])
        categorical_cols = self.feature_groups.get('categorical', {}).get('columns', [])
        
        numeric_indices = []
        categorical_indices = []
        
        for i, col in enumerate(self.feature_columns):
            if col in numeric_cols:
                numeric_indices.append(i)
            elif col in categorical_cols:
                categorical_indices.append(i)
        
        if numeric_indices:
            numeric_features = sequences[:, :, numeric_indices]
        else:
            batch_size, seq_len = sequences.shape[0], sequences.shape[1]
            numeric_features = torch.zeros(batch_size, seq_len, 0, device=sequences.device)
        
        categorical_features = {}
        for i, col in enumerate(categorical_cols):
            if col in self.feature_columns:
                col_idx = self.feature_columns.index(col)
                categorical_features[col] = sequences[:, :, col_idx].long()
        
        return numeric_features, categorical_features
        
    def create_dataloaders(self, window_size=8, seed=42):
        
        batch_size = self.hyperparams['batch_size']
        use_cuda = torch.cuda.is_available()
        
        
        train_dataset = SlidingWindowDataset(self.X_train, self.y_train, window_size, seed)
        val_dataset = SlidingWindowDataset(self.X_val, self.y_val, window_size, seed + 1) 
        test_dataset = SlidingWindowDataset(self.X_test, self.y_test, window_size, seed + 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=1,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=1,
            drop_last=False
        )
        
        class EpochAwareDataLoader:
            def __init__(self, dataloader, dataset):
                self.dataloader = dataloader
                self.dataset = dataset
                self.epoch = 0
                
            def __iter__(self):
                if hasattr(self.dataset, 'reshuffle'):
                    self.dataset.reshuffle(self.epoch + 42)
                    self.epoch += 1
                return iter(self.dataloader)
            
            def __len__(self):
                return len(self.dataloader)
        
        train_loader_wrapped = EpochAwareDataLoader(train_loader, train_dataset)
        val_loader_wrapped = EpochAwareDataLoader(val_loader, val_dataset)
        test_loader_wrapped = EpochAwareDataLoader(test_loader, test_dataset)
        
        logger.info(f"DataLoaders simulando RandomSlidingWindowSampler creati:")
        logger.info(f"  Window size: {window_size}")
        logger.info(f"  Workers: train=8, val/test=4")
        logger.info(f"  Multiprocessing: ABILITATO")
        logger.info(f"  Train: {len(train_loader)} batch")
        logger.info(f"  Val: {len(val_loader)} batch") 
        logger.info(f"  Test: {len(test_loader)} batch")
        
        return train_loader_wrapped, val_loader_wrapped, test_loader_wrapped

def save_model_transformer(model, hyperparams, feature_columns, filepath, 
                          n_classes, class_mapping, feature_groups, vocab_sizes=None, 
                          class_weights=None, temporal_config=None):
    """Salva il modello Transformer e la configurazione"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'feature_columns': feature_columns,
        'feature_groups': feature_groups,
        'n_classes': n_classes,
        'class_mapping': class_mapping,
        'vocab_sizes': vocab_sizes,
        'class_weights': class_weights,
        'temporal_config': temporal_config,
        'model_type': 'temporal_transformer'
    }, filepath)
    logger.info(f"Modello Transformer salvato in: {filepath}")


def load_model_transformer(filepath, device='cpu'):
    """Carica il modello Transformer salvato"""
    checkpoint = torch.load(filepath, map_location=device)
    
    if checkpoint.get('model_type') != 'temporal_transformer':
        logger.warning("Questo non sembra essere un modello Transformer temporale!")
    
    model = NetworkTrafficTransformer(
        checkpoint['hyperparams'], 
        checkpoint['n_classes'],
        checkpoint['feature_groups'],
        checkpoint.get('vocab_sizes', {})
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint