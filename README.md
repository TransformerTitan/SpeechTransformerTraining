# Memory-Efficient Gradient Accumulation for Speech Transformer Training

A PyTorch implementation of adaptive gradient accumulation strategies for training speech transformers with limited GPU memory, based on the research paper "Memory-Efficient Gradient Accumulation for Speech Transformer Training: A Practical Approach" by P. Yadla.

## 🎯 Overview

Training large speech transformers like Wav2Vec2 and Whisper requires substantial GPU memory (16-40GB+), creating barriers for researchers with limited resources. This implementation provides:

- **35-42% memory reduction** with minimal performance impact (0.1-0.3% WER increase)
- **Adaptive batching** that considers variable sequence lengths in speech data
- **Mixed precision training** support for additional memory savings
- **Consumer GPU compatibility** - train larger models on RTX 3090/4090 class hardware

## 🚀 Key Features

### Adaptive Batch Construction
- **Length-aware batching**: Automatically adjusts batch sizes based on sequence length distribution
- **Memory estimation**: Real-time memory usage prediction to prevent OOM errors
- **Variable sequence handling**: Optimized for speech data's irregular sequence lengths (1-30 seconds)

### Memory Optimization Techniques
- **Gradient accumulation**: Simulate large batch training with multiple forward passes
- **Mixed precision (FP16)**: Reduce memory usage by ~40% with automatic loss scaling
- **Smart memory allocation**: Dynamic batch sizing based on available GPU memory

### Production-Ready Features
- **Configurable memory limits**: Set maximum GPU memory usage
- **Progress monitoring**: Real-time memory usage and training metrics
- **Error handling**: Graceful handling of memory constraints and edge cases

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/TransformerTitan/SpeechTransformerTraining.git
cd memory-efficient-speech-training

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy librosa  # For audio processing
pip install transformers datasets  # For Wav2Vec2/Whisper models

# Optional: Install for development
pip install -e .
```

## 🔧 Quick Start

### Basic Usage

```python
from memory_efficient_trainer import MemoryEfficientTrainer, TrainingConfig
import torch.nn as nn

# Configure training parameters
config = TrainingConfig(
    max_memory_gb=12.0,        # Your GPU memory limit
    accumulation_steps=4,       # Gradient accumulation steps
    learning_rate=1e-4,
    mixed_precision=True,       # Enable FP16
    adaptive_batching=True      # Enable adaptive batch sizing
)

# Initialize your model (example with Wav2Vec2)
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Create trainer
trainer = MemoryEfficientTrainer(model, config)

# Train with your dataset
avg_loss = trainer.train_epoch(your_speech_dataset)
```

### Advanced Configuration

```python
# Fine-tune memory usage parameters
config = TrainingConfig(
    max_memory_gb=8.0,                    # Conservative memory limit
    accumulation_steps=8,                 # More accumulation for smaller batches
    learning_rate=5e-5,                   # Lower LR for stability
    mixed_precision=True,
    adaptive_batching=True,
    sequence_length_percentile=90.0,      # Memory estimation percentile
    alpha_constant=1.5                    # Architecture-specific constant
)
```

## 📊 Performance Results

Based on LibriSpeech experiments with Wav2Vec2 models:

| Method | Peak Memory | Memory Reduction | WER Impact | Training Speed |
|--------|-------------|------------------|------------|----------------|
| Standard Training | 18.2 GB | — | 2.3% | 1.00× |
| Adaptive Accumulation | 10.6 GB | **42%** | 2.4% (+0.1%) | 0.82× |
| Adaptive + Mixed Precision | 8.1 GB | **55%** | 2.6% (+0.3%) | 0.91× |

## 🛠️ Integration with Popular Models

### Wav2Vec2 Training

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset

# Load model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Configure for large model
config = TrainingConfig(
    max_memory_gb=16.0,  # Wav2Vec2-Large requires more memory
    accumulation_steps=6,
    mixed_precision=True
)

trainer = MemoryEfficientTrainer(model, config)
```

### Whisper Fine-tuning

```python
from transformers import WhisperForConditionalGeneration

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Whisper-specific configuration
config = TrainingConfig(
    max_memory_gb=14.0,
    accumulation_steps=4,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    mixed_precision=True
)

trainer = MemoryEfficientTrainer(model, config)
```

## 📋 Configuration Options

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_memory_gb` | float | 12.0 | Maximum GPU memory to use (GB) |
| `accumulation_steps` | int | 4 | Number of gradient accumulation steps |
| `learning_rate` | float | 1e-4 | Learning rate for optimizer |
| `mixed_precision` | bool | True | Enable automatic mixed precision (FP16) |
| `adaptive_batching` | bool | True | Enable length-aware batch construction |
| `sequence_length_percentile` | float | 95.0 | Percentile for memory estimation |
| `alpha_constant` | float | 1.2 | Architecture-dependent constant |

## 🔍 Memory Estimation

The trainer uses the memory estimation formula from the paper:

```
Memory = Model(θ) + Σ Activation(Ti) + Gradients(θ)
Activation(Ti) ≈ α × Ti × d_model × n_layers
```

Where:
- `Ti` = sequence length for sample i
- `α` = architecture-dependent constant (default: 1.2)
- `d_model` = model hidden dimension
- `n_layers` = number of transformer layers

## 📈 Monitoring and Debugging

### Memory Usage Tracking

```python
import torch

# Monitor GPU memory during training
if torch.cuda.is_available():
    print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Training Logs

The trainer provides detailed logging:
```
INFO - Model memory: 1.42 GB
INFO - Max memory limit: 12.00 GB
INFO - Batch 10, Loss: 2.3456, Batch size: 8, GPU Memory: 9.34 GB
INFO - Epoch completed in 245.67s, Average loss: 2.1234
```

## ⚡ Hardware Recommendations

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- **Models**: Wav2Vec2-Base, Whisper-Base
- **Expected Memory Reduction**: 30-35%

### Recommended Setup
- **GPU**: 16-24GB VRAM (RTX 3090, RTX 4090, RTX A5000)
- **Models**: Wav2Vec2-Large, Whisper-Large
- **Expected Memory Reduction**: 40-45%

### High-End Configuration
- **GPU**: 32GB+ VRAM (RTX 6000 Ada, A100)
- **Models**: Custom large models, multi-task training
- **Expected Memory Reduction**: 45-55%

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Errors**
```python
# Reduce memory limit and increase accumulation steps
config = TrainingConfig(
    max_memory_gb=6.0,  # Lower limit
    accumulation_steps=8,  # More accumulation
    mixed_precision=True
)
```

**Slow Training Speed**
```python
# Balance memory vs speed
config = TrainingConfig(
    max_memory_gb=14.0,  # Use more memory
    accumulation_steps=2,  # Less accumulation
    mixed_precision=True
)
```

**Memory Estimation Issues**
```python
# Adjust architecture constant for your model
config = TrainingConfig(
    alpha_constant=1.5,  # Increase for conservative estimation
    sequence_length_percentile=90.0  # Use lower percentile
)
```

## 📚 Implementation Details

### Adaptive Batching Algorithm

The core batching algorithm (Algorithm 1 from paper):

1. **Sort sequences** by length for efficient packing
2. **Estimate memory** for each potential batch addition
3. **Add sequences** while memory constraint is satisfied
4. **Yield batch** when memory limit reached

### Gradient Accumulation Strategy

Implements the gradient accumulation formula:
```
g_acc = (1/K) × Σ(k=1 to K) g_k
```

Where gradients are normalized by accumulation steps (K) rather than total samples for stable learning dynamics.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone for development
git clone [<repository-url>](https://github.com/TransformerTitan/SpeechTransformerTraining)

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Check code formatting
black memory_efficient_trainer.py
flake8 memory_efficient_trainer.py
```

## 📝 Citation

TBD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper by P. Yadla
- Hugging Face Transformers team for model implementations
- PyTorch team for automatic mixed precision support
- Speech processing community for valuable feedback


---

**Made with ❤️ for the speech processing community**
