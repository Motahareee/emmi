import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class VAEEncoder(nn.Module):
    """
    VAE encoder — edge-side only.

    Maps the fused multimodal embedding to a latent distribution (μ, log σ²).
    At inference, only μ is transmitted to save bandwidth.

    Input:  [B, d_in]
    Output: (mu [B, d_latent], log_var [B, d_latent])   during training
             mu [B, d_latent]                            during inference
    """

    def __init__(self, d_in: int, d_latent: int = 64,
                 hidden_dims: tuple = (256, 128)):
        super().__init__()

        layers = []
        in_dim = d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        self.fc_mu      = nn.Linear(in_dim, d_latent)
        self.fc_log_var = nn.Linear(in_dim, d_latent)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10, 4)   # prevent exp overflow / total collapse
        return mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path — returns μ only (no sampling, no log_var)."""
        mu, _ = self.forward(x)
        return mu

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class VAEDecoder(nn.Module):
    """
    VAE decoder — server-side only.

    Reconstructs the shared multimodal embedding from the latent vector,
    which is then projected into the MLLM's token space.

    Input:  [B, d_latent]
    Output: [B, d_out]  (should match d_shared from CrossModalAlignment)
    """

    def __init__(self, d_latent: int = 64, d_out: int = 256,
                 hidden_dims: tuple = (128, 256)):
        super().__init__()

        layers = []
        in_dim = d_latent
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_out))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    """
    Full VAE (encoder + decoder) used for standalone pre-training.

    After training, call vae.save(path). The edge and server pipelines
    then load only the half they need via VAE.load(path).

    Input:  [B, d_in]    (fused multimodal embedding from Stage 2)
    Output: (recon [B, d_in], mu [B, d_latent], log_var [B, d_latent])
    """

    def __init__(self, d_in: int, d_latent: int = 64,
                 encoder_hidden_dims: tuple = (256, 128),
                 decoder_hidden_dims: tuple = (128, 256)):
        super().__init__()
        self.encoder = VAEEncoder(d_in, d_latent, encoder_hidden_dims)
        self.decoder = VAEDecoder(d_latent, d_in, decoder_hidden_dims)
        # Store dims so we can reconstruct the model on load
        self._d_in                  = d_in
        self._d_latent              = d_latent
        self._encoder_hidden_dims   = encoder_hidden_dims
        self._decoder_hidden_dims   = decoder_hidden_dims

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = VAEEncoder.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":           self.state_dict(),
            "d_in":                 self._d_in,
            "d_latent":             self._d_latent,
            "encoder_hidden_dims":  self._encoder_hidden_dims,
            "decoder_hidden_dims":  self._decoder_hidden_dims,
        }, path)

    @classmethod
    def load(cls, path: str) -> "VAE":
        ckpt = torch.load(path, map_location="cpu")
        vae  = cls(
            d_in=ckpt["d_in"],
            d_latent=ckpt["d_latent"],
            encoder_hidden_dims=ckpt["encoder_hidden_dims"],
            decoder_hidden_dims=ckpt["decoder_hidden_dims"],
        )
        vae.load_state_dict(ckpt["state_dict"])
        return vae


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0, free_bits: float = 0.5) -> dict:
    """
    β-VAE loss with free-bits regularisation to prevent posterior collapse.

    Loss = recon_MSE + β * sum_d max(KL_d, free_bits)

    free_bits: minimum KL nats per latent dimension.  Dimensions below this
               threshold get zero KL gradient, letting reconstruction loss
               push them to encode real information before regularisation
               pressure kicks in.  Typical values: 0.5 – 2.0 nats/dim.

    Args:
        recon      : decoder output [B, d_shared]
        target     : original fused embedding [B, d_shared]
        mu         : encoder mean [B, d_latent]
        log_var    : encoder log variance [B, d_latent]
        beta       : KL weight (use warmup scheduler in train.py)
        free_bits  : min KL per dimension before penalty applies

    Returns dict with 'loss', 'recon_loss', 'kl_loss' for logging.
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # Per-dimension KL: [B, d_latent] → mean over batch → [d_latent]
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = kl_per_dim.mean(0)                          # [d_latent]
    kl_loss    = kl_per_dim.clamp(min=free_bits).mean()      # scalar

    loss = recon_loss + beta * kl_loss
    return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
