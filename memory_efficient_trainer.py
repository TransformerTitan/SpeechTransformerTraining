import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for memory-efficient training"""
    max_memory_gb: float = 12.0  # Maximum GPU memory to use
    accumulation_steps: int = 4
    learning_rate: float = 1e-4
    mixed_precision: bool = True
    adaptive_batching: bool = True
    sequence_length_percentile: float = 95.0  # For memory estimation
    alpha_constant: float = 1.2  # Architecture-dependent constant from paper


class SpeechDataset(Dataset):
    """Mock speech dataset with variable sequence lengths"""
    def __init__(self, num_samples: int = 1000, min_length: int = 16000, max_length: int = 480000):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        
        # Generate realistic sequence length distribution (log-normal)
        self.sequence_lengths = np.random.lognormal(
            mean=np.log(80000), sigma=0.8, size=num_samples
        ).astype(int)
        self.sequence_lengths = np.clip(self.sequence_lengths, min_length, max_length)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        length = self.sequence_lengths[idx]
        # Mock audio features (mel-spectrograms or raw waveform)
        audio = torch.randn(length, 80)  # 80-dim features
        # Mock labels for speech recognition
        label_length = max(1, length // 320)  # Rough alignment ratio
        labels = torch.randint(0, 1000, (label_length,))  # Vocab size 1000
        
        return {
            'audio': audio,
            'labels': labels,
            'audio_length': length,
            'label_length': label_length
        }


class MemoryEstimator:
    """Estimates memory usage for transformer models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_memory_gb = 0.0
        self.cached_activations = {}
        
    def estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate base model memory in GB"""
        total_params = sum(p.numel() for p in model.parameters())
        # Parameters + gradients + optimizer states (Adam: 2x params)
        bytes_per_param = 4 if not self.config.mixed_precision else 2
        total_bytes = total_params * bytes_per_param * 4  # params + grads + 2x optimizer
        return total_bytes / (1024**3)  # Convert to GB
    
    def estimate_activation_memory(self, sequence_length: int, d_model: int, n_layers: int) -> float:
        """Estimate activation memory for a sequence (Equation 2 from paper)"""
        if sequence_length in self.cached_activations:
            return self.cached_activations[sequence_length]
        
        # Activation(Ti) ≈ α · Ti · d_model · n_layers
        bytes_per_element = 4 if not self.config.mixed_precision else 2
        activation_elements = self.config.alpha_constant * sequence_length * d_model * n_layers
        activation_bytes = activation_elements * bytes_per_element
        activation_gb = activation_bytes / (1024**3)
        
        self.cached_activations[sequence_length] = activation_gb
        return activation_gb
    
    def estimate_batch_memory(self, batch_lengths: List[int], d_model: int, n_layers: int) -> float:
        """Estimate total memory for a batch"""
        total_activation_memory = sum(
            self.estimate_activation_memory(length, d_model, n_layers) 
            for length in batch_lengths
        )
        return self.model_memory_gb + total_activation_memory


class AdaptiveBatchSampler:
    """Length-aware batch construction (Algorithm 1 from paper)"""
    
    def __init__(self, dataset: SpeechDataset, memory_estimator: MemoryEstimator, 
                 config: TrainingConfig, d_model: int, n_layers: int):
        self.dataset = dataset
        self.memory_estimator = memory_estimator
        self.config = config
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Sort sequences by length for efficient batching
        self.sorted_indices = sorted(
            range(len(dataset)), 
            key=lambda i: dataset.sequence_lengths[i]
        )
        
    def __iter__(self):
        """Generate batches with adaptive size based on memory constraints"""
        i = 0
        while i < len(self.sorted_indices):
            batch_indices = []
            batch_lengths = []
            
            while i < len(self.sorted_indices):
                idx = self.sorted_indices[i]
                length = self.dataset.sequence_lengths[idx]
                
                # Estimate memory if we add this sequence
                test_lengths = batch_lengths + [length]
                estimated_memory = self.memory_estimator.estimate_batch_memory(
                    test_lengths, self.d_model, self.n_layers
                )
                
                if estimated_memory <= self.config.max_memory_gb:
                    batch_indices.append(idx)
                    batch_lengths.append(length)
                    i += 1
                else:
                    break
            
            if batch_indices:  # Ensure we have at least one sample
                yield batch_indices
            else:
                # If even a single sample exceeds memory, yield it anyway
                if i < len(self.sorted_indices):
                    yield [self.sorted_indices[i]]
                    i += 1


class MemoryEfficientTrainer:
    """Main trainer with adaptive gradient accumulation"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize memory estimator
        self.memory_estimator = MemoryEstimator(config)
        self.memory_estimator.model_memory_gb = self.memory_estimator.estimate_model_memory(model)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Mixed precision setup
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Model memory: {self.memory_estimator.model_memory_gb:.2f} GB")
        logger.info(f"Max memory limit: {config.max_memory_gb:.2f} GB")
    
    def collate_fn(self, batch):
        """Custom collate function for variable-length sequences"""
        # Sort batch by length for efficient packing
        batch = sorted(batch, key=lambda x: x['audio_length'], reverse=True)
        
        # Pad sequences
        max_audio_len = batch[0]['audio_length']
        max_label_len = max(item['label_length'] for item in batch)
        
        padded_audio = torch.zeros(len(batch), max_audio_len, batch[0]['audio'].size(-1))
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        audio_lengths = torch.tensor([item['audio_length'] for item in batch])
        label_lengths = torch.tensor([item['label_length'] for item in batch])
        
        for i, item in enumerate(batch):
            padded_audio[i, :item['audio_length']] = item['audio']
            padded_labels[i, :item['label_length']] = item['labels']
        
        return {
            'audio': padded_audio,
            'labels': padded_labels,
            'audio_lengths': audio_lengths,
            'label_lengths': label_lengths
        }
    
    def compute_loss(self, batch):
        """Compute loss for a batch (mock implementation)"""
        audio = batch['audio'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Mock forward pass - replace with actual model
        # For demonstration, we'll use a simple linear layer
        batch_size, seq_len, feat_dim = audio.shape
        output = self.model(audio.view(-1, feat_dim))
        output = output.view(batch_size, seq_len, -1)
        
        # Mock CTC loss or cross-entropy
        loss = nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), 
            labels.view(-1), 
            ignore_index=0
        )
        
        return loss
    
    def train_epoch(self, dataset: SpeechDataset, d_model: int = 768, n_layers: int = 12):
        """Train for one epoch with adaptive gradient accumulation"""
        self.model.train()
        
        # Create adaptive batch sampler
        if self.config.adaptive_batching:
            batch_sampler = AdaptiveBatchSampler(
                dataset, self.memory_estimator, self.config, d_model, n_layers
            )
        else:
            # Fixed batch size fallback
            batch_sampler = [list(range(i, min(i+8, len(dataset)))) 
                           for i in range(0, len(dataset), 8)]
        
        total_loss = 0.0
        num_batches = 0
        accumulation_count = 0
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        start_time = time.time()
        
        for batch_indices in batch_sampler:
            # Create batch from indices
            batch_data = [dataset[i] for i in batch_indices]
            batch = self.collate_fn(batch_data)
            
            # Forward pass with optional mixed precision
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(batch)
                
                # Scale loss for gradient accumulation (Equation 3 from paper)
                scaled_loss = loss / self.config.accumulation_steps
                self.scaler.scale(scaled_loss).backward()
            else:
                loss = self.compute_loss(batch)
                scaled_loss = loss / self.config.accumulation_steps
                scaled_loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            accumulation_count += 1
            
            # Update parameters after accumulation_steps
            if accumulation_count >= self.config.accumulation_steps:
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
                
                # Log progress
                if num_batches % 10 == 0:
                    avg_batch_size = len(batch_indices)
                    memory_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
                    logger.info(f"Batch {num_batches}, Loss: {loss.item():.4f}, "
                              f"Batch size: {avg_batch_size}, GPU Memory: {memory_used:.2f} GB")
        
        # Handle remaining gradients
        if accumulation_count > 0:
            if self.config.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        logger.info(f"Epoch completed in {epoch_time:.2f}s, Average loss: {avg_loss:.4f}")
        return avg_loss


# Mock model for demonstration
class MockSpeechTransformer(nn.Module):
    """Mock speech transformer for demonstration"""
    def __init__(self, input_dim: int = 80, hidden_dim: int = 768, num_classes: int = 1000):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.dropout(torch.relu(self.linear2(x)))
        return self.linear3(x)


def main():
    """Example usage of the memory-efficient training system"""
    
    # Configuration
    config = TrainingConfig(
        max_memory_gb=8.0,  # Adjust based on your GPU
        accumulation_steps=4,
        learning_rate=1e-4,
        mixed_precision=True,
        adaptive_batching=True
    )
    
    # Create mock model and dataset
    model = MockSpeechTransformer()
    dataset = SpeechDataset(num_samples=100)  # Small dataset for demo
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(model, config)
    
    # Train for one epoch
    logger.info("Starting memory-efficient training...")
    avg_loss = trainer.train_epoch(dataset)
    
    logger.info(f"Training completed. Final average loss: {avg_loss:.4f}")
    
    # Memory usage summary
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak GPU memory usage: {max_memory:.2f} GB")


if __name__ == "__main__":
    main()
