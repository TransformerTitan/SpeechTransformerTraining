"""Tests for memory estimation functionality."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, Mock

from speech_transformer_training.trainer import MemoryEstimator, TrainingConfig


class TestMemoryEstimator:
    """Test cases for MemoryEstimator class."""
    
    def test_init(self, training_config):
        """Test MemoryEstimator initialization."""
        estimator = MemoryEstimator(training_config)
        assert estimator.config == training_config
        assert estimator.model_memory_gb == 0.0
        assert estimator.cached_activations == {}
    
    def test_estimate_model_memory_small_model(self, training_config, simple_model):
        """Test model memory estimation for small model."""
        estimator = MemoryEstimator(training_config)
        memory_gb = estimator.estimate_model_memory(simple_model)
        
        # Should be a reasonable small value
        assert 0.001 < memory_gb < 0.1
        assert isinstance(memory_gb, float)
    
    def test_estimate_model_memory_mixed_precision(self, simple_model):
        """Test model memory estimation with mixed precision."""
        config_fp16 = TrainingConfig(mixed_precision=True)
        config_fp32 = TrainingConfig(mixed_precision=False)
        
        estimator_fp16 = MemoryEstimator(config_fp16)
        estimator_fp32 = MemoryEstimator(config_fp32)
        
        memory_fp16 = estimator_fp16.estimate_model_memory(simple_model)
        memory_fp32 = estimator_fp32.estimate_model_memory(simple_model)
        
        # FP16 should use less memory
        assert memory_fp16 < memory_fp32
        assert memory_fp16 / memory_fp32 == pytest.approx(0.5, rel=0.1)
    
    def test_estimate_activation_memory_basic(self, memory_estimator):
        """Test basic activation memory estimation."""
        sequence_length = 5000
        d_model = 768
        n_layers = 12
        
        memory_gb = memory_estimator.estimate_activation_memory(
            sequence_length, d_model, n_layers
        )
        
        # Should be positive and reasonable
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
        
        # Should follow the formula: α * Ti * d_model * n_layers
        expected_elements = (
            memory_estimator.config.alpha_constant * 
            sequence_length * d_model * n_layers
        )
        bytes_per_element = 4 if not memory_estimator.config.mixed_precision else 2
        expected_gb = (expected_elements * bytes_per_element) / (1024**3)
        
        assert memory_gb == pytest.approx(expected_gb, rel=0.01)
    
    def test_estimate_activation_memory_caching(self, memory_estimator):
        """Test that activation memory estimation is cached."""
        sequence_length = 5000
        d_model = 768
        n_layers = 12
        
        # First call
        memory1 = memory_estimator.estimate_activation_memory(
            sequence_length, d_model, n_layers
        )
        
        # Second call should use cache
        memory2 = memory_estimator.estimate_activation_memory(
            sequence_length, d_model, n_layers
        )
        
        assert memory1 == memory2
        assert sequence_length in memory_estimator.cached_activations
    
    def test_estimate_activation_memory_scaling(self, memory_estimator):
        """Test that activation memory scales with sequence length."""
        d_model = 768
        n_layers = 12
        
        short_memory = memory_estimator.estimate_activation_memory(1000, d_model, n_layers)
        long_memory = memory_estimator.estimate_activation_memory(10000, d_model, n_layers)
        
        # Memory should scale linearly with sequence length
        assert long_memory / short_memory == pytest.approx(10.0, rel=0.01)
    
    def test_estimate_batch_memory_single_sequence(self, memory_estimator, simple_model):
        """Test batch memory estimation with single sequence."""
        memory_estimator.model_memory_gb = memory_estimator.estimate_model_memory(simple_model)
        
        batch_lengths = [5000]
        d_model = 64  # From simple_model
        n_layers = 2   # From simple_model (2 linear layers)
        
        total_memory = memory_estimator.estimate_batch_memory(
            batch_lengths, d_model, n_layers
        )
        
        expected_activation = memory_estimator.estimate_activation_memory(
            5000, d_model, n_layers
        )
        expected_total = memory_estimator.model_memory_gb + expected_activation
        
        assert total_memory == pytest.approx(expected_total, rel=0.01)
    
    def test_estimate_batch_memory_multiple_sequences(self, memory_estimator, simple_model):
        """Test batch memory estimation with multiple sequences."""
        memory_estimator.model_memory_gb = memory_estimator.estimate_model_memory(simple_model)
        
        batch_lengths = [2000, 5000, 8000, 3000]
        d_model = 64
        n_layers = 2
        
        total_memory = memory_estimator.estimate_batch_memory(
            batch_lengths, d_model, n_layers
        )
        
        # Should be model memory + sum of activation memories
        expected_activations = sum(
            memory_estimator.estimate_activation_memory(length, d_model, n_layers)
            for length in batch_lengths
        )
        expected_total = memory_estimator.model_memory_gb + expected_activations
        
        assert total_memory == pytest.approx(expected_total, rel=0.01)
    
    def test_estimate_batch_memory_empty_batch(self, memory_estimator, simple_model):
        """Test batch memory estimation with empty batch."""
        memory_estimator.model_memory_gb = memory_estimator.estimate_model_memory(simple_model)
        
        total_memory = memory_estimator.estimate_batch_memory([], 768, 12)
        
        # Should equal model memory only
        assert total_memory == memory_estimator.model_memory_gb
    
    def test_mixed_precision_effect(self):
        """Test that mixed precision reduces memory estimates."""
        config_fp32 = TrainingConfig(mixed_precision=False)
        config_fp16 = TrainingConfig(mixed_precision=True)
        
        estimator_fp32 = MemoryEstimator(config_fp32)
        estimator_fp16 = MemoryEstimator(config_fp16)
        
        # Test activation memory
        memory_fp32 = estimator_fp32.estimate_activation_memory(5000, 768, 12)
        memory_fp16 = estimator_fp16.estimate_activation_memory(5000, 768, 12)
        
        # FP16 should use half the memory
        assert memory_fp16 == pytest.approx(memory_fp32 / 2, rel=0.01)
    
    def test_alpha_constant_effect(self):
        """Test that alpha constant affects memory estimation."""
        config1 = TrainingConfig(alpha_constant=1.0)
        config2 = TrainingConfig(alpha_constant=2.0)
        
        estimator1 = MemoryEstimator(config1)
        estimator2 = MemoryEstimator(config2)
        
        memory1 = estimator1.estimate_activation_memory(5000, 768, 12)
        memory2 = estimator2.estimate_activation_memory(5000, 768, 12)
        
        # Memory should scale with alpha constant
        assert memory2 == pytest.approx(memory1 * 2, rel=0.01)
    
    @pytest.mark.parametrize("sequence_length,d_model,n_layers", [
        (1000, 512, 6),
        (5000, 768, 12),
        (15000, 1024, 24),
        (30000, 768, 12),
    ])
    def test_estimate_activation_memory_parameters(self, memory_estimator, 
                                                 sequence_length, d_model, n_layers):
        """Test activation memory estimation with various parameters."""
        memory_gb = memory_estimator.estimate_activation_memory(
            sequence_length, d_model, n_layers
        )
        
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
        
        # Verify it follows the expected formula
        expected_elements = (
            memory_estimator.config.alpha_constant * 
            sequence_length * d_model * n_layers
        )
        bytes_per_element = 4 if not memory_estimator.config.mixed_precision else 2
        expected_gb = (expected_elements * bytes_per_element) / (1024**3)
        
        assert memory_gb == pytest.approx(expected_gb, rel=0.01)
    
    def test_memory_estimation_consistency(self, memory_estimator):
        """Test that memory estimation is consistent across calls."""
        params = (5000, 768, 12)
        
        # Multiple calls should return same result
        results = [
            memory_estimator.estimate_activation_memory(*params)
            for _ in range(5)
        ]
        
        assert all(r == results[0] for r in results)
