"""
Plain autoencoder for embedding compression — no KL term, pure reconstruction.

512-dim CLIP embeddings → 64-dim bottleneck → 512-dim reconstruction.
Trained with MSE loss only.  Encoder runs on edge, decoder on server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class AEEncoder(nn.Module):
    """Edge-side: compresses 512-dim CLIP embedding to 64-dim."""

    def __init__(self, d_in: int = 512, d_latent: int = 64,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)    # [B, d_latent]


class AEDecoder(nn.Module):
    """Server-side: reconstructs 512-dim embedding from 64-dim latent."""

    def __init__(self, d_latent: int = 64, d_out: int = 512,
                 hidden_dims: tuple = (128, 256, 512)):
        super().__init__()
        layers, in_dim = [], d_latent
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)    # [B, d_out]


class AutoEncoder(nn.Module):
    """Full autoencoder used for training."""

    def __init__(self, d_in: int = 512, d_latent: int = 64,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        self.encoder = AEEncoder(d_in, d_latent, hidden_dims)
        self.decoder = AEDecoder(d_latent, d_in, tuple(reversed(hidden_dims)))
        self._d_in      = d_in
        self._d_latent  = d_latent
        self._hidden    = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  self.state_dict(),
            "d_in":        self._d_in,
            "d_latent":    self._d_latent,
            "hidden_dims": self._hidden,
        }, path)

    @classmethod
    def load(cls, path: str) -> "AutoEncoder":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ae   = cls(ckpt["d_in"], ckpt["d_latent"], ckpt["hidden_dims"])
        ae.load_state_dict(ckpt["state_dict"])
        return ae


# ── Loss functions ─────────────────────────────────────────────────────────────

def supcon_loss(latents: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al. 2020) on AE latents.

    Pulls samples with the same label together in latent space and pushes
    samples with different labels apart — without using labels at deployment.

    Args:
        latents     : [B, d_latent]  AE encoder outputs
        labels      : [B]            binary match labels (0 / 1)
        temperature : scalar         softmax temperature (default 0.07)

    Returns:
        scalar loss (0 if no valid positive pairs in batch)
    """
    B = latents.shape[0]
    z = F.normalize(latents, dim=1)           # unit-norm latents [B, d_latent]

    # Pairwise cosine similarity scaled by temperature  [B, B]
    sim = (z @ z.T) / temperature

    # Numerical stability: subtract row-wise max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Positive mask: same label, excluding self  [B, B]
    lbl      = labels.view(-1)
    pos_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0.)

    # Denominator: sum over all pairs except self
    self_mask = torch.eye(B, dtype=torch.bool, device=latents.device)
    exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Per-anchor loss averaged over its positives
    log_prob = sim - log_denom                              # [B, B]
    n_pos    = pos_mask.sum(dim=1)                          # [B]
    loss_per = -(pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

    # Only count anchors that have at least one positive pair
    valid = n_pos > 0
    if not valid.any():
        return torch.tensor(0.0, device=latents.device)
    return loss_per[valid].mean()

def cross_modal_infonce_loss(z: torch.Tensor,
                             text_emb: torch.Tensor,
                             img_emb: torch.Tensor,
                             text_proj: nn.Linear,
                             img_proj: nn.Linear,
                             temperature: float = 0.07) -> torch.Tensor:
    """
    Cross-modal InfoNCE loss for label-free compressor training.

    Encourages the 64-dim latent z to be a sufficient statistic for
    cross-modal alignment by maximising mutual information between z
    and each individual modality embedding.

    Positive pairs are (z_i, text_i) and (z_i, img_i) — the natural
    modality pairing, available without binary match labels.

    Args:
        z         : [B, d_latent]  AE encoder output
        text_emb  : [B, d_enc]    text encoder embedding (frozen)
        img_emb   : [B, d_enc]    image encoder embedding (frozen)
        text_proj : Linear(d_enc → d_latent)  learnable projection head
        img_proj  : Linear(d_enc → d_latent)  learnable projection head
        temperature : softmax temperature

    Returns:
        scalar loss
    """
    z_n = F.normalize(z, dim=1)
    t_n = F.normalize(text_proj(text_emb), dim=1)
    v_n = F.normalize(img_proj(img_emb),  dim=1)

    labels = torch.arange(len(z), device=z.device)

    # InfoNCE: latent should match its own modality embedding (on-diagonal)
    loss_t = F.cross_entropy((z_n @ t_n.T) / temperature, labels)
    loss_v = F.cross_entropy((z_n @ v_n.T) / temperature, labels)
    return (loss_t + loss_v) / 2


def _cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix. [B, D] → [B, B]"""
    x_norm = F.normalize(x, dim=1)
    return x_norm @ x_norm.T


def distilae_loss(recon: torch.Tensor, target: torch.Tensor,
                  latents: torch.Tensor,
                  lambda_distil: float = 0.1,
                  mask_diag: bool = False,
                  distil_only: bool = False) -> dict:
    """
    DistilAE loss: MSE reconstruction + similarity matrix distillation.
    Encourages the latent space to preserve pairwise cosine similarity
    structure of the original CLIP embeddings.

    Args:
        mask_diag   : exclude the diagonal (always 1.0 vs 1.0, trivially
                      matched) from the similarity MSE so the loss is driven
                      entirely by cross-pair structure.
        distil_only : drop the reconstruction term; train purely on
                      similarity preservation.
    """
    recon_loss  = F.mse_loss(recon, target)
    S_orig      = _cosine_sim_matrix(target)
    S_lat       = _cosine_sim_matrix(latents)
    if mask_diag:
        off = ~torch.eye(len(S_orig), dtype=torch.bool, device=S_orig.device)
        distil_loss = F.mse_loss(S_lat[off], S_orig[off])
    else:
        distil_loss = F.mse_loss(S_lat, S_orig)
    if distil_only:
        loss = distil_loss
    else:
        loss = recon_loss + lambda_distil * distil_loss
    return {"loss": loss, "recon_loss": recon_loss, "distil_loss": distil_loss}


def distvarae_loss(recon: torch.Tensor, target: torch.Tensor,
                   latents: torch.Tensor,
                   lambda_distil: float = 0.1,
                   lambda_var: float = 0.001) -> dict:
    """
    DistilVarAE loss: MSE + similarity distillation + variance maximization.
    Adds a term that encourages latents to be spread out (high variance),
    preventing collapse and improving discriminability.
    """
    recon_loss  = F.mse_loss(recon, target)
    S_orig      = _cosine_sim_matrix(target)
    S_lat       = _cosine_sim_matrix(latents)
    distil_loss = F.mse_loss(S_lat, S_orig)
    var_loss    = -latents.var(dim=0).mean()   # maximize variance = minimize negative
    loss        = recon_loss + lambda_distil * distil_loss + lambda_var * var_loss
    return {"loss": loss, "recon_loss": recon_loss,
            "distil_loss": distil_loss, "var_loss": var_loss}
