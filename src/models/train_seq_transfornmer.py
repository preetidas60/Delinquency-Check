# Minimal Transformer skeleton for sequence learning. Requires sequence datasets.
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.out = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: batch x seq_len x input_dim
        x = self.input_proj(x)
        x = x.permute(1,0,2)  # seq_len x batch x d_model
        x = self.encoder(x)
        x = x.mean(dim=0)
        return torch.sigmoid(self.out(x)).squeeze(-1)
