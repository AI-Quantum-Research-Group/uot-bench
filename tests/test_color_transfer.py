import pytest

pytest.importorskip("PIL", reason="color-transfer extra not installed")
pytest.importorskip("skimage", reason="color-transfer extra not installed")

import numpy as np
import jax.numpy as jnp

from uot.experiments.real_data.color_transfer.utils import (
    im2mat,
    mat2im,
    match_shape,
)
import uot.experiments.real_data.color_transfer.color_transfer_metrics as ct_metrics


class TestImageUtilities:
    """Test basic image manipulation utilities"""
    
    def test_im2mat_basic(self):
        """Test image to matrix conversion"""
        img = np.random.rand(10, 15, 3)
        mat = im2mat(img)
        assert mat.shape == (150, 3)
        
    def test_mat2im_basic(self):
        """Test matrix to image conversion"""
        mat = np.random.rand(150, 3)
        shape = (10, 15, 3)
        img = mat2im(mat, shape)
        assert img.shape == shape
        
    def test_im2mat_mat2im_roundtrip(self):
        """Test that im2mat and mat2im are inverses"""
        original_img = np.random.rand(8, 12, 3)
        mat = im2mat(original_img)
        reconstructed_img = mat2im(mat, original_img.shape)
        assert np.allclose(original_img, reconstructed_img)
        
    def test_match_shape_upscaling(self):
        """Test image upscaling when target is larger"""
        source_img = np.random.rand(20, 30, 3)
        target_img = np.random.rand(40, 60, 3)
        
        resized = match_shape(target_img, source_img)
        assert resized.shape[:2] == (20, 30)
        
    def test_match_shape_downscaling(self):
        """Test image downscaling when target is smaller"""
        source_img = np.random.rand(40, 60, 3)
        target_img = np.random.rand(20, 30, 3)
        
        resized = match_shape(target_img, source_img)
        assert resized.shape[:2] == (40, 60)


class TestColorTransferMetrics:
    """Test color transfer metrics functions"""
    
    def setup_method(self):
        """Set up test images"""
        self.img1 = np.random.rand(50, 50, 3)
        self.img2 = np.random.rand(50, 50, 3)
        
    def test_compute_ssim_metric_range(self):
        """Test SSIM metric returns values in valid range"""
        ssim_val = ct_metrics.compute_ssim_metric(self.img1, self.img2)
        assert -1 <= ssim_val <= 1
        
    def test_compute_ssim_metric_identical_images(self):
        """Test SSIM returns 1 for identical images"""
        ssim_val = ct_metrics.compute_ssim_metric(self.img1, self.img1)
        assert np.isclose(ssim_val, 1.0)
        
    def test_compute_delta_e_positive(self):
        """Test Delta E returns positive values"""
        delta_e = ct_metrics.compute_delta_e(self.img1, self.img2)
        assert delta_e >= 0
        
    def test_compute_delta_e_identical_images(self):
        """Test Delta E returns 0 for identical images"""
        delta_e = ct_metrics.compute_delta_e(self.img1, self.img1)
        assert np.isclose(delta_e, 0.0, atol=1e-10)
        
    def test_compute_colorfulness_positive(self):
        """Test colorfulness metric returns positive values"""
        colorfulness = ct_metrics.compute_colorfulness(self.img1)
        assert colorfulness >= 0
        
    def test_rgb2gray_shape(self):
        """Test RGB to grayscale conversion"""
        gray = ct_metrics.rgb2gray(self.img1)
        assert gray.shape == self.img1.shape[:2]
        
    def test_compute_gradient_magnitude_correlation_range(self):
        """Test gradient correlation returns values in valid range"""
        corr = ct_metrics.compute_gradient_magnitude_correlation(self.img1, self.img2)
        assert -1 <= corr <= 1 or np.isnan(corr)
        
    def test_compute_laplacian_variance_positive(self):
        """Test Laplacian variance returns non-negative values"""
        lap_var = ct_metrics.compute_laplacian_variance(self.img1)
        assert lap_var >= 0


class TestMetricsEdgeCases:
    """Test edge cases in metrics"""
    
    def test_gradient_correlation_flat_images(self):
        """Test gradient correlation with flat (constant) images"""
        flat_img = np.ones((50, 50, 3)) * 0.5
        corr = ct_metrics.compute_gradient_magnitude_correlation(flat_img, flat_img)
        assert corr == 0.0
        
    def test_colorfulness_grayscale_image(self):
        """Test colorfulness with grayscale-like image"""
        gray_img = np.random.rand(50, 50, 1)
        gray_img = np.repeat(gray_img, 3, axis=2)
        
        colorfulness = ct_metrics.compute_colorfulness(gray_img)
        assert colorfulness >= 0