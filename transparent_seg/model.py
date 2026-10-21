"""MiT-B0 encoder + SegFormer-style MLP fusion + optional boundary head.

The decoder is initialized from scratch; no ADE20K segmentation weights.
The auxiliary head is not executed for inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig, SegformerModel


class TransparentSegformer(nn.Module):
    def __init__(self, boundary=False, pretrained=True, revision="main", cache_dir=None):
        super().__init__()
        config = SegformerConfig(
            depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
            num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
            decoder_hidden_size=256, reshape_last_stage=True,
        )
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(
                "nvidia/mit-b0", revision=revision, cache_dir=cache_dir)
        else:
            self.encoder = SegformerModel(config)
        if self.encoder.config.hidden_sizes != config.hidden_sizes:
            raise ValueError("Expected MiT-B0 architecture")
        self.encoder_revision = getattr(self.encoder.config, "_commit_hash", None) or revision
        # Pointwise projections are mathematically equivalent to per-pixel MLPs.
        self.projections = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in config.hidden_sizes])
        self.fuse = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                  nn.BatchNorm2d(256), nn.ReLU())
        self.dropout = nn.Dropout(0.1)
        self.segmentation = nn.Conv2d(256, 1, 1)
        # Created last so shared A/B/C/D parameters initialize identically per seed.
        self.boundary = (nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(64, 1, 1)) if boundary else None)

    def forward(self, images, return_boundary=False):
        features = self.encoder(images, output_hidden_states=True).hidden_states
        size = features[0].shape[-2:]
        projected = [F.interpolate(layer(x), size=size, mode="bilinear", align_corners=False)
                     for layer, x in zip(self.projections, features)]
        fused = self.fuse(torch.cat(projected[::-1], dim=1))
        output = {"logits": self.segmentation(self.dropout(fused))}
        if return_boundary:
            if self.boundary is None:
                raise ValueError("Boundary output requested from baseline")
            output["boundary_logits"] = self.boundary(fused)
        return output
