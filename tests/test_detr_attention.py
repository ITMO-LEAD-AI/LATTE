import torch
from latte.detr_attention import DetrLATTEAttention


def test_detr_original_reshape_and_state_dict():
    module = DetrLATTEAttention(32, 4)
    x, p = torch.randn(2, 7, 32), torch.randn(2, 7, 32)
    shape = (2, 7, 4, 8)
    q = module.q_proj(x + p).view(shape).transpose(1, 2)
    k = module.k_proj(x + p).view(shape).transpose(1, 2)
    v = module.v_proj(x).view(shape).transpose(1, 2)
    raw, _ = module.linear_attention_forward_qtvit(
        q, k, v, None, module.scaling, ones_scale1=module.ones_scale1)
    expected = module.o_proj(raw.reshape(2, 7, 32).contiguous())
    actual, weights = module(x, position_embeddings=p)
    torch.testing.assert_close(actual, expected)
    assert weights is None
    assert "ones_scale1" in module.state_dict()
    assert "ones_scale" not in module.state_dict()
