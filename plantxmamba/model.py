import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from .mamba import MambaModule


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=5, emb_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_size)

    def forward(self, x):
        B, C, _, _ = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return self.proj(x.view(B, -1, C * self.patch_size * self.patch_size))


class PlantXMamba(nn.Module):
    def __init__(self, num_classes, patch_size=5, emb_size=16, d_state=64, d_conv=64, expand=4,
                 n_layers=2, num_blocks=4, dropout=0.1, pretrained=True):
        super().__init__()
        # First two VGG16 blocks: (B, 3, 224, 224) -> (B, 128, 56, 56)
        vgg = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        self.vgg_block = nn.Sequential(*list(vgg.features[:10]))
        self.patch_embed = PatchEmbedding(128, patch_size, emb_size)
        args = type("Args", (), dict(d_model=emb_size, d_state=d_state, d_conv=d_conv,
                                     expand=expand, n_layers=n_layers, dropout=dropout))()
        self.mamba = nn.Sequential(*[MambaModule(args) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(emb_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(self.vgg_block(x))
        x = self.norm(self.mamba(x)).permute(0, 2, 1)
        return self.classifier(self.global_pool(x).squeeze(-1))
