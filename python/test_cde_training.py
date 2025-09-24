#!/usr/bin/env python3
"""
Test script for conditional density estimation training.
This script validates MDN and MAF implementations.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from cde_training import MDNEstimator, MAFEstimator, generate_test_data
import autograd.numpy as anp


def test_mdn_basic():
    """Test basic MDN functionality."""
    print("Testing MDN basic functionality...")
    
    # Create a simple MDN
    mdn = MDNEstimator(param_dim=2, feature_dim=1, n_components=3, hidden_sizes=(16,))
    
    # Generate simple test data
    params, features = generate_test_data('banana', n_samples=100, seed=42)
    
    # Train briefly
    mdn.train(params, features, n_iter=50, learning_rate=1e-3, use_tqdm=False)
    
    # Test sampling
    rng = anp.random.RandomState(42)
    test_features = anp.array([[1.0]])
    samples = mdn.sample(test_features, 10, rng)
    
    assert samples.shape == (1, 10, 2), f"Expected shape (1, 10, 2), got {samples.shape}"
    assert not anp.any(anp.isnan(samples)), "Samples contain NaN values"
    
    # Test log probability
    log_probs = mdn.log_prob(features[:10], params[:10])
    assert log_probs.shape == (10,), f"Expected shape (10,), got {log_probs.shape}"
    assert anp.all(anp.isfinite(log_probs)), "Log probabilities contain non-finite values"
    
    print("✓ MDN basic functionality test passed")


def test_maf_basic():
    """Test basic MAF functionality."""
    print("Testing MAF basic functionality...")
    
    # Create a simple MAF
    maf = MAFEstimator(param_dim=2, feature_dim=1, n_flows=2, hidden_units=16)
    
    # Generate simple test data
    params, features = generate_test_data('moons', n_samples=100, seed=42)
    
    # Train briefly
    maf.train(params, features, n_iter=50, learning_rate=1e-3, use_tqdm=False)
    
    # Test sampling
    rng = anp.random.RandomState(42)
    test_features = anp.array([[0.1]])
    samples = maf.sample(test_features, 10, rng)
    
    assert samples.shape == (1, 10, 2), f"Expected shape (1, 10, 2), got {samples.shape}"
    assert not anp.any(anp.isnan(samples)), "Samples contain NaN values"
    
    # Test log probability
    log_probs = maf.log_prob(features[:10], params[:10])
    assert log_probs.shape == (10,), f"Expected shape (10,), got {log_probs.shape}"
    assert anp.all(anp.isfinite(log_probs)), "Log probabilities contain non-finite values"
    
    print("✓ MAF basic functionality test passed")


def test_data_generation():
    """Test data generation functions."""
    print("Testing data generation...")
    
    datasets = ['banana', 'student_t', 'moons']
    
    for dataset in datasets:
        params, features = generate_test_data(dataset, n_samples=50, seed=42)
        
        assert params.shape == (50, 2), f"Expected params shape (50, 2), got {params.shape}"
        assert features.shape == (50, 1), f"Expected features shape (50, 1), got {features.shape}"
        assert not anp.any(anp.isnan(params)), f"Params contain NaN for dataset {dataset}"
        assert not anp.any(anp.isnan(features)), f"Features contain NaN for dataset {dataset}"
    
    print("✓ Data generation test passed")


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    # Test invalid dimensions
    try:
        mdn = MDNEstimator(param_dim=0, feature_dim=1)
        assert False, "Should raise ValueError for invalid param_dim"
    except ValueError:
        pass
    
    try:
        maf = MAFEstimator(param_dim=1, feature_dim=-1)
        assert False, "Should raise ValueError for invalid feature_dim"
    except ValueError:
        pass
    
    # Test training without weights
    mdn = MDNEstimator(param_dim=2, feature_dim=1)
    try:
        test_features = anp.array([[1.0]])
        test_params = anp.array([[1.0, 2.0]])
        mdn.log_prob(test_features, test_params)
        assert False, "Should raise RuntimeError when model not trained"
    except RuntimeError:
        pass
    
    print("✓ Error handling test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running CDE Training Tests")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_mdn_basic()
        test_maf_basic()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)