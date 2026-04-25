import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    Encode a short reference video into a sequence of ``dim``-dim tokens that can
    be concatenated to T5 prompt embeddings and fed into LongLive's cross-attention.

    The VAE is shared with the generator and always frozen; only the temporal
    transformer and the patch/positional projections are trained.

    Input:  motion_video [B, T_ref, 3, H, W], values in [-1, 1]
    Output: motion_tokens [B, L_mot, dim] where L_mot = T_lat * tokens_per_frame
    """

    def __init__(
        self,
        vae_wrapper,
        dim: int = 4096,
        num_layers: int = 4,
        num_heads: int = 16,
        tokens_per_frame: int = 64,
        max_tokens: int = 512,
        latent_channels: int = 16,
    ):
        super().__init__()
        for p in vae_wrapper.parameters():
            p.requires_grad_(False)
        # Stash the VAE in a list so it is NOT registered as a submodule:
        # motion_encoder is FSDP-wrapped and we don't want the VAE re-sharded
        # or its frozen params dragged into state_dict.
        object.__setattr__(self, "_vae_ref", [vae_wrapper])

        self.dim = dim
        self.tokens_per_frame = tokens_per_frame

        self.patch_proj = nn.Linear(latent_channels, dim)
        self.pool_queries = nn.Parameter(torch.randn(1, tokens_per_frame, dim) * 0.02)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    @property
    def vae(self):
        return self._vae_ref[0]

    @torch.no_grad()
    def _encode_vae(self, motion_video: torch.Tensor) -> torch.Tensor:
        pixel = motion_video.permute(0, 2, 1, 3, 4).contiguous()
        return self.vae.encode_to_latent(pixel)

    def forward(self, motion_video: torch.Tensor) -> torch.Tensor:
        latent = self._encode_vae(motion_video)
        B, T_lat, C, h, w = latent.shape

        flat = latent.reshape(B * T_lat, C, h * w).permute(0, 2, 1)
        flat = flat.to(self.patch_proj.weight.dtype)
        proj = self.patch_proj(flat)

        queries = self.pool_queries.expand(B * T_lat, -1, -1)
        pooled, _ = self.attention_pool(query=queries, key=proj, value=proj, need_weights=False)
        tokens = pooled.reshape(B, T_lat * self.tokens_per_frame, self.dim)

        L_mot = tokens.shape[1]
        assert L_mot <= self.pos_embed.shape[1], \
            f"L_mot={L_mot} exceeds max_tokens={self.pos_embed.shape[1]}"
        tokens = tokens + self.pos_embed[:, :L_mot] + self.type_embed

        tokens = self.transformer(tokens)
        return self.out_norm(tokens)
