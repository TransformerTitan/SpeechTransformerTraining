"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Import from the main package (adjust import path as needed)
try:
    from speech_transformer_training.trainer import TrainingConfig, MemoryEstimator
    from speech_transformer_training.dataset import SpeechDataset
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from speech_transformer_training.trainer import TrainingConfig, MemoryEstimator
    from speech_transformer_training.dataset import SpeechDataset


@pytest.fixture
def device():
    """Fixture for device selection."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def training_config():
    """Default training configuration for tests."""
    return TrainingConfig(
        max_memory_gb=4.0,  # Conservative for testing
        accumulation_steps=2,
        learning_rate=1e-4,
        mixed_precision=False,  # Disable for deterministic tests
        adaptive_batching=True,
        sequence_length_percentile=95.0,
        alpha_constant=1.2
    )


@pytest.fixture
def small_training_config():
    """Small training configuration for fast tests."""
    return TrainingConfig(
        max_memory_gb=2.0,
        accumulation_steps=2,
        learning_rate=1e-3,
        mixed_precision=False,
        adaptive_batching=True
    )


@pytest.fixture
def mock_model():
    """Mock PyTorch model for testing."""
    model = Mock(spec=torch.nn.Module)
    model.parameters.return_value = [
        torch.randn(100, 50, requires_grad=True),
        torch.randn(50, requires_grad=True),
        torch.randn(50, 10, requires_grad=True),
        torch.randn(10, requires_grad=True),
    ]
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None
    return model


@pytest.fixture
def simple_model():
    """Simple real PyTorch model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=80, hidden_dim=64, output_dim=100):
            super().__init__()
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.1)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.dropout(x)
            return self.linear2(x)
    
    return SimpleModel()


@pytest.fixture
def speech_dataset():
    """Small speech dataset for testing."""
    return SpeechDataset(
        num_samples=20,
        min_length=1000,
        max_length=10000
    )


@pytest.fixture
def variable_length_sequences():
    """Generate sequences with variable lengths for testing."""
    lengths = [1000, 2500, 5000, 7500, 10000, 15000, 20000]
    sequences = []
    for length in lengths:
        sequences.append({
            'audio': torch.randn(length, 80),
            'labels': torch.randint(0, 100, (length // 10,)),
            'audio_length': length,
            'label_length': length // 10
        })
    return sequences


@pytest.fixture
def memory_estimator(training_config):
    """Memory estimator fixture."""
    return MemoryEstimator(training_config)


@pytest.fixture
def batch_data():
    """Sample batch data for testing."""
    return {
        'audio': torch.randn(4, 5000, 80),
        'labels': torch.randint(0, 100, (4, 500)),
        'audio_lengths': torch.tensor([5000, 4800, 4500, 3000]),
        'label_lengths': torch.tensor([500, 480, 450, 300])
    }


@pytest.fixture
def mock_cuda():
    """Mock CUDA functions for testing without GPU."""
    with pytest.MonkeyPatch().context() as m:
        # Mock torch.cuda functions
        m.setattr(torch.cuda, 'is_available', lambda: True)
        m.setattr(torch.cuda, 'memory_allocated', lambda: 1024**3)  # 1GB
        m.setattr(torch.cuda, 'max_memory_allocated', lambda: 2 * 1024**3)  # 2GB
        m.setattr(torch.cuda, 'memory_reserved', lambda: 3 * 1024**3)  # 3GB
        m.setattr(torch.cuda, 'empty_cache', lambda: None)
        yield


@pytest.fixture
def sample_audio_features():
    """Sample audio features for different sequence lengths."""
    return {
        'short': torch.randn(1000, 80),
        'medium': torch.randn(5000, 80),
        'long': torch.randn(15000, 80),
        'very_long': torch.randn(30000, 80)
    }


@pytest.fixture
def gradient_accumulation_data():
    """Data for testing gradient accumulation."""
    return {
        'gradients': [
            torch.randn(100, 50),
            torch.randn(100, 50),
            torch.randn(100, 50),
            torch.randn(100, 50),
        ],
        'losses': [2.5, 2.3, 2.1, 2.0],
        'batch_sizes': [4, 6, 8, 5]
    }


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.param_groups = [{'lr': 1e-4}]
    return optimizer


@pytest.fixture
def mock_scaler():
    """Mock gradient scaler for mixed precision testing."""
    scaler = Mock()
    scaler.scale = Mock(return_value=Mock())
    scaler.step = Mock()
    scaler.update = Mock()
    return scaler


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sequence_length_distributions():
    """Different sequence length distributions for testing."""
    return {
        'uniform': np.random.uniform(1000, 10000, 100).astype(int),
        'normal': np.clip(np.random.normal(5000, 2000, 100), 1000, 15000).astype(int),
        'exponential': np.clip(np.random.exponential(3000, 100), 1000, 20000).astype(int),
        'bimodal': np.concatenate([
            np.random.normal(2000, 500, 50),
            np.random.normal(8000, 1000, 50)
        ]).astype(int)
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add gpu marker to tests that require GPU
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add slow marker to tests that are slow
        if "slow" in item.nodeid.lower() or "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
