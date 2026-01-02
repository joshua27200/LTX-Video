import torch

from ltx_video.pipelines.crf_compressor import compress


class TestCRFCompressor:
    """Test cases for the CRF compressor module."""

    def test_compress_even_dimensions(self):
        """Test compression with even dimensions (no padding needed)."""
        image = torch.rand(100, 100, 3)
        result = compress(image, crf=29)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_compress_odd_dimensions_preserves_shape(self):
        """Test that compression with odd dimensions preserves the original shape."""
        image = torch.rand(101, 99, 3)
        result = compress(image, crf=29)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_compress_odd_height_even_width(self):
        """Test compression with odd height and even width."""
        image = torch.rand(101, 100, 3)
        result = compress(image, crf=29)

        assert result.shape == image.shape

    def test_compress_even_height_odd_width(self):
        """Test compression with even height and odd width."""
        image = torch.rand(100, 101, 3)
        result = compress(image, crf=29)

        assert result.shape == image.shape

    def test_compress_crf_zero_bypass(self):
        """Test that crf=0 bypasses compression and returns the same tensor."""
        image = torch.rand(101, 99, 3)
        result = compress(image, crf=0)

        assert result is image

    def test_compress_preserves_device(self):
        """Test that compression preserves the device of the tensor."""
        image = torch.rand(100, 100, 3)
        result = compress(image, crf=29)

        assert result.device == image.device

    def test_compress_output_range(self):
        """Test that output values are in the expected range [0, 1]."""
        image = torch.rand(100, 100, 3)
        result = compress(image, crf=29)

        assert result.min() >= 0.0
        assert result.max() <= 1.0
