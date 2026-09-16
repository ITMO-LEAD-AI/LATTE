import torch

from latte.attention import RTDetrLATTEAttention


def test_rtdetr_attention_shape_with_positions():
    module = RTDetrLATTEAttention(hidden_size=32, num_attention_heads=4)
    inputs = torch.randn(2, 30, 32)
    positions = torch.randn(2, 30, 32)
    output, weights = module(inputs, position_embeddings=positions)
    assert output.shape == inputs.shape
    assert weights is None
    assert torch.isfinite(output).all()
