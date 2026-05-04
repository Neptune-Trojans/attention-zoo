import torch
import torch.nn.functional as F

from attention.self_attention import SelfAttention


def test_output_shape_matches_input():
    torch.manual_seed(42)
    batch_size, seq_length, input_dim = 4, 10, 64
    x = torch.randn(batch_size, seq_length, input_dim)

    model = SelfAttention(input_dim)
    model.eval()

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape


def test_attention_weights_sum_to_one():
    torch.manual_seed(0)
    input_dim = 16
    x = torch.randn(2, 5, input_dim)

    model = SelfAttention(input_dim)
    queries = model.query(x)
    keys = model.key(x)
    scores = torch.bmm(queries, keys.transpose(1, 2)) / (input_dim ** 0.5)
    weights = F.softmax(scores, dim=2)

    row_sums = weights.sum(dim=2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_gradients_flow_to_all_projections():
    torch.manual_seed(0)
    input_dim = 8
    x = torch.randn(1, 4, input_dim, requires_grad=True)

    model = SelfAttention(input_dim)
    output = model(x)
    output.sum().backward()

    assert model.query.weight.grad is not None
    assert model.key.weight.grad is not None
    assert model.value.weight.grad is not None
    assert (model.query.weight.grad.abs().sum() > 0).item()
    assert (model.key.weight.grad.abs().sum() > 0).item()
    assert (model.value.weight.grad.abs().sum() > 0).item()


def test_batch_independence():
    torch.manual_seed(0)
    input_dim = 16
    model = SelfAttention(input_dim)
    model.eval()

    x1 = torch.randn(1, 6, input_dim)
    x2 = torch.randn(1, 6, input_dim)
    batched = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        out_batched = model(batched)

    assert torch.allclose(out_batched[0:1], out1, atol=1e-6)
    assert torch.allclose(out_batched[1:2], out2, atol=1e-6)


def test_output_shape_matches_input_4d():
    torch.manual_seed(42)
    outer, inner, seq_length, input_dim = 2, 3, 5, 16
    x = torch.randn(outer, inner, seq_length, input_dim)

    model = SelfAttention(input_dim)
    model.eval()

    with torch.no_grad():
        output = model(x)

    assert output.shape == x.shape


def test_4d_matches_looped_3d():
    torch.manual_seed(0)
    outer, inner, seq_length, input_dim = 2, 3, 5, 16
    x = torch.randn(outer, inner, seq_length, input_dim)

    model = SelfAttention(input_dim)
    model.eval()

    with torch.no_grad():
        out_4d = model(x)
        out_looped = torch.stack([model(x[i]) for i in range(outer)], dim=0)

    assert torch.allclose(out_4d, out_looped, atol=1e-6)


def test_attention_weights_sum_to_one_4d():
    torch.manual_seed(0)
    input_dim = 16
    x = torch.randn(2, 3, 5, input_dim)

    model = SelfAttention(input_dim)
    queries = model.query(x)
    keys = model.key(x)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / (input_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
