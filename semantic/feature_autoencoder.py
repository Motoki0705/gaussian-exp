import torch
from torch import nn


class FeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim=384, latent_dim=3, hidden_dim=256):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        z_flat = self.encoder(x_flat)
        return z_flat.reshape(*original_shape[:-1], self.latent_dim)

    def decode(self, z):
        original_shape = z.shape
        z_flat = z.reshape(-1, original_shape[-1])
        x_flat = self.decoder(z_flat)
        return x_flat.reshape(*original_shape[:-1], self.in_dim)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
