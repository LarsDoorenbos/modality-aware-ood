
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
  
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)
    

class SelfTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.v_to_a_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.a_to_v_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_a1 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.v_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.a_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)

    def forward(self, v, a):
        """
        v: (B, Tv, D)
        a: (B, Ta, D)
        """

        # Cross-attention
        v_attn, _ = self.v_to_a_attn(
            query=self.norm_v1(v),
            key=self.norm_a1(a),
            value=self.norm_a1(a),
        )
        a_attn, _ = self.a_to_v_attn(
            query=self.norm_a1(a),
            key=self.norm_v1(v),
            value=self.norm_v1(v),
        )

        v = v + v_attn
        a = a + a_attn

        # Feed-forward
        v = v + self.v_mlp(self.norm_v2(v))
        a = a + self.a_mlp(self.norm_a2(a))

        return v, a


class BigEncoder(nn.Module):
    def __init__(
        self,
        v_dim: int = 2048,
        a_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 8,
        num_heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()

        # Project modalities into shared embedding space
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.a_proj = nn.Linear(a_dim, hidden_dim)

        # Stacked cross-modal Transformer blocks
        self.blocks = nn.ModuleList([
            CrossModalTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, vfeat, afeat):
        """
        vfeat: (B, Tv, v_dim) or (B, v_dim)
        afeat: (B, Ta, a_dim) or (B, a_dim)
        """

        # Handle vector inputs
        if vfeat.dim() == 2:
            vfeat = vfeat.unsqueeze(1)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)

        v = self.v_proj(vfeat)
        a = self.a_proj(afeat)

        for block in self.blocks:
            v, a = block(v, a)

        # Early fusion via pooling
        v = v.mean(dim=1)
        a = a.mean(dim=1)

        fused = self.norm(v + a)

        return self.classifier(fused)
    
    def get_features(self, vfeat, afeat):
        """
        vfeat: (B, Tv, v_dim) or (B, v_dim)
        afeat: (B, Ta, a_dim) or (B, a_dim)
        """

        # Handle vector inputs
        if vfeat.dim() == 2:
            vfeat = vfeat.unsqueeze(1)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)

        v = self.v_proj(vfeat)
        a = self.a_proj(afeat)

        for block in self.blocks:
            v, a = block(v, a)

        # Early fusion via pooling
        v = v.mean(dim=1)
        a = a.mean(dim=1)

        fused = self.norm(v + a)

        return fused


class MidEncoder(nn.Module):
    def __init__(
        self,
        v_dim=2048,
        a_dim=768,
        hidden_dim=512,
        out_dim=8,
        num_heads=8,
        depth=4,
    ):
        super().__init__()

        assert depth % 2 == 0, "depth must be even for mid-fusion"

        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.a_proj = nn.Linear(a_dim, hidden_dim)

        half = depth // 2

        # Stage 1: modality-specific self-attention
        self.v_self_blocks = nn.ModuleList([
            SelfTransformerBlock(hidden_dim, num_heads)
            for _ in range(half)
        ])
        self.a_self_blocks = nn.ModuleList([
            SelfTransformerBlock(hidden_dim, num_heads)
            for _ in range(half)
        ])

        # Stage 2: cross-modal fusion
        self.cross_blocks = nn.ModuleList([
            CrossModalTransformerBlock(hidden_dim, num_heads)
            for _ in range(half)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, vfeat, afeat):
        """
        vfeat: (B, Tv, v_dim) or (B, v_dim)
        afeat: (B, Ta, a_dim) or (B, a_dim)
        """

        if vfeat.dim() == 2:
            vfeat = vfeat.unsqueeze(1)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)

        v = self.v_proj(vfeat)
        a = self.a_proj(afeat)

        # Stage 1: independent processing
        for vb, ab in zip(self.v_self_blocks, self.a_self_blocks):
            v = vb(v)
            a = ab(a)

        # Stage 2: fusion
        for block in self.cross_blocks:
            v, a = block(v, a)

        # Pool and classify
        v = v.mean(dim=1)
        a = a.mean(dim=1)

        fused = self.norm(v + a)
        return self.classifier(fused)
    
    def get_features(self, vfeat, afeat):
        """
        vfeat: (B, Tv, v_dim) or (B, v_dim)
        afeat: (B, Ta, a_dim) or (B, a_dim)
        """

        if vfeat.dim() == 2:
            vfeat = vfeat.unsqueeze(1)
        if afeat.dim() == 2:
            afeat = afeat.unsqueeze(1)

        v = self.v_proj(vfeat)
        a = self.a_proj(afeat)

        # Stage 1: independent processing
        for vb, ab in zip(self.v_self_blocks, self.a_self_blocks):
            v = vb(v)
            a = ab(a)

        # Stage 2: fusion
        for block in self.cross_blocks:
            v, a = block(v, a)

        # Pool and classify
        v = v.mean(dim=1)
        a = a.mean(dim=1)

        fused = self.norm(v + a)
        return fused
