import torch

from attention.cross_attention import CrossAttention


def test_output_shape_matches_query():
    torch.manual_seed(42)
    batch, seq_q, seq_kv, query_dim, kv_dim = 4, 10, 7, 64, 32
    x_q = torch.randn(batch, seq_q, query_dim)
    x_k = torch.randn(batch, seq_kv, kv_dim)
    x_v = torch.randn(batch, seq_kv, kv_dim)

    model = CrossAttention(query_dim, kv_dim)
    model.eval()

    with torch.no_grad():
        result = model(x_q, x_k, x_v)

    assert result.output.shape == x_q.shape


def test_attention_weights_shape_and_sum():
    torch.manual_seed(0)
    batch, seq_q, seq_kv, query_dim, kv_dim = 2, 5, 8, 16, 24
    x_q = torch.randn(batch, seq_q, query_dim)
    x_k = torch.randn(batch, seq_kv, kv_dim)
    x_v = torch.randn(batch, seq_kv, kv_dim)

    model = CrossAttention(query_dim, kv_dim)
    with torch.no_grad():
        result = model(x_q, x_k, x_v)

    assert result.attention_weights.shape == (batch, seq_q, seq_kv)
    row_sums = result.attention_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_gradients_flow_to_all_projections():
    torch.manual_seed(0)
    query_dim, kv_dim = 8, 12
    x_q = torch.randn(1, 4, query_dim, requires_grad=True)
    x_k = torch.randn(1, 6, kv_dim, requires_grad=True)
    x_v = torch.randn(1, 6, kv_dim, requires_grad=True)

    model = CrossAttention(query_dim, kv_dim)
    result = model(x_q, x_k, x_v)
    result.output.sum().backward()

    assert model.query.weight.grad is not None
    assert model.key.weight.grad is not None
    assert model.value.weight.grad is not None
    assert (model.query.weight.grad.abs().sum() > 0).item()
    assert (model.key.weight.grad.abs().sum() > 0).item()
    assert (model.value.weight.grad.abs().sum() > 0).item()


def test_batch_independence():
    torch.manual_seed(0)
    query_dim, kv_dim = 16, 16
    model = CrossAttention(query_dim, kv_dim)
    model.eval()

    x_q1, x_k1, x_v1 = torch.randn(1, 6, query_dim), torch.randn(1, 4, kv_dim), torch.randn(1, 4, kv_dim)
    x_q2, x_k2, x_v2 = torch.randn(1, 6, query_dim), torch.randn(1, 4, kv_dim), torch.randn(1, 4, kv_dim)
    x_q_b = torch.cat([x_q1, x_q2], dim=0)
    x_k_b = torch.cat([x_k1, x_k2], dim=0)
    x_v_b = torch.cat([x_v1, x_v2], dim=0)

    with torch.no_grad():
        out1 = model(x_q1, x_k1, x_v1).output
        out2 = model(x_q2, x_k2, x_v2).output
        out_b = model(x_q_b, x_k_b, x_v_b).output

    assert torch.allclose(out_b[0:1], out1, atol=1e-6)
    assert torch.allclose(out_b[1:2], out2, atol=1e-6)


def test_output_shape_matches_query_4d():
    torch.manual_seed(0)
    outer, inner = 2, 3
    seq_q, seq_kv, query_dim, kv_dim = 5, 4, 16, 8
    x_q = torch.randn(outer, inner, seq_q, query_dim)
    x_k = torch.randn(outer, inner, seq_kv, kv_dim)
    x_v = torch.randn(outer, inner, seq_kv, kv_dim)

    model = CrossAttention(query_dim, kv_dim)
    model.eval()

    with torch.no_grad():
        result = model(x_q, x_k, x_v)

    assert result.output.shape == x_q.shape
    assert result.attention_weights.shape == (outer, inner, seq_q, seq_kv)


def test_4d_matches_looped_3d():
    torch.manual_seed(0)
    outer = 2
    seq_q, seq_kv, query_dim, kv_dim = 5, 4, 16, 8
    x_q = torch.randn(outer, seq_q, query_dim)
    x_k = torch.randn(outer, seq_kv, kv_dim)
    x_v = torch.randn(outer, seq_kv, kv_dim)

    model = CrossAttention(query_dim, kv_dim)
    model.eval()

    with torch.no_grad():
        out_batched = model(x_q, x_k, x_v).output
        out_looped = torch.stack(
            [model(x_q[i:i+1], x_k[i:i+1], x_v[i:i+1]).output.squeeze(0) for i in range(outer)],
            dim=0,
        )

    assert torch.allclose(out_batched, out_looped, atol=1e-6)
