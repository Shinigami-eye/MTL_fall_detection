"""Tests for windowing and preprocessing."""

import numpy as np
import pytest

from preprocessing import WindowGenerator, Normalizer


class TestWindowGenerator:
    """Test window generation functionality."""
    
    def test_window_generation(self):
        """Test basic window generation."""
        # Create sample data
        data = np.random.randn(200, 6)  # 200 samples, 6 channels
        labels = np.zeros(200)
        labels[100:] = 1  # Second half has different label
        
        # Create window generator
        gen = WindowGenerator(window_size=50, stride=25, sampling_rate=50)
        
        # Generate windows
        windows = gen.generate_windows(data, labels)
        
        # Check number of windows
        expected_windows = (200 - 50) // 25 + 1
        assert len(windows) == expected_windows
        
        # Check window properties
        for window in windows:
            assert window['data'].shape == (50, 6)
            assert 'label' in window
            assert 'start_idx' in window
            assert 'end_idx' in window
    
    def test_label_assignment(self):
        """Test majority voting for label assignment."""
        # Create data with clear label boundaries
        data = np.random.randn(100, 3)
        labels = np.array([0] * 30 + [1] * 40 + [0] * 30)
        
        gen = WindowGenerator(window_size=20, stride=10)
        windows = gen.generate_windows(data, labels)
        
        # Check that middle windows have label 1
        middle_window = windows[len(windows) // 2]
        assert middle_window['label'] == 1
    
    def test_magnitude_channel(self):
        """Test magnitude channel computation."""
        gen = WindowGenerator(window_size=10, stride=5)
        
        # Create 3D vector data
        data = np.array([[3, 4, 0], [0, 0, 5], [1, 1, 1]])
        
        # Compute magnitude
        magnitude = gen._compute_magnitude(data)
        
        expected = np.array([5.0, 5.0, np.sqrt(3)])
        np.testing.assert_allclose(magnitude, expected)


class TestNormalizer:
    """Test normalization functionality."""
    
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        # Create sample data
        data = np.random.randn(1000, 6) * 10 + 5
        
        # Create normalizer
        normalizer = Normalizer()
        
        # Fit and transform
        normalized = normalizer.fit_transform(data)
        
        # Check that data is normalized
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-6)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        # Create sample data
        original = np.random.randn(100, 3) * 5 + 10
        
        normalizer = Normalizer()
        normalized = normalizer.fit_transform(original)
        recovered = normalizer.inverse_transform(normalized)
        
        # Check that inverse transform recovers original data
        np.testing.assert_allclose(recovered, original, rtol=1e-5)
    
    def test_save_load_stats(self, tmp_path):
        """Test saving and loading normalization statistics."""
        # Create and fit normalizer
        data = np.random.randn(100, 3)
        normalizer1 = Normalizer(stats_path=tmp_path / "stats.json")
        normalizer1.fit(data)
        
        # Create new normalizer and load stats
        normalizer2 = Normalizer(stats_path=tmp_path / "stats.json")
        normalizer2.load_stats()
        
        # Check that stats are the same
        assert normalizer1.stats == normalizer2.stats
        assert normalizer2.fitted