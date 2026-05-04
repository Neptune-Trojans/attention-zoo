# Self-Attention
### Reference

Vaswani et al. (2017) - "Attention Is All You Need"
https://arxiv.org/abs/1706.03762

![Self-Attention Mechanism](images/self_attention.png)

### Usage

```python
import torch
from attention.self_attention import SelfAttention

# Create model
model = SelfAttention(input_dim=64)

# Input shape: (batch_size, group, seq_length, input_dim)
x = torch.randn(2, 3, 10, 64)

# Forward pass
output = model(x)  # Output shape: (2, 3, 10, 64)
```




