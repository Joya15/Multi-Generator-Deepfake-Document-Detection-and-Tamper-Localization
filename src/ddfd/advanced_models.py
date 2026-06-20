from __future__ import annotations

from typing import Any


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _ensure_nchw(x):
    if x.ndim == 4 and x.shape[-1] in {32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024}:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def _ensure_nchw_channels(x, channels: int):
    if x.ndim == 4:
        if x.shape[1] == channels:
            return x
        for dim in (2, 3):
            if x.shape[dim] == channels:
                return x.movedim(dim, 1).contiguous()
    return _ensure_nchw(x)


class TorchvisionBackboneSeg:
    def __new__(cls, backbone_name: str, dropout: float = 0.1, pretrained: bool = False):
        torch, nn, F = _torch()
        import torchvision.models as tvm

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if backbone_name == "convnext_tiny":
                    weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                    backbone = tvm.convnext_tiny(weights=weights)
                    out_ch = 768
                    self.features = backbone.features
                elif backbone_name == "convnext_small":
                    weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
                    backbone = tvm.convnext_small(weights=weights)
                    out_ch = 768
                    self.features = backbone.features
                elif backbone_name == "swin_t":
                    weights = tvm.Swin_T_Weights.DEFAULT if pretrained else None
                    backbone = tvm.swin_t(weights=weights)
                    out_ch = 768
                    self.features = backbone.features
                elif backbone_name == "swin_s":
                    weights = tvm.Swin_S_Weights.DEFAULT if pretrained else None
                    backbone = tvm.swin_s(weights=weights)
                    out_ch = 768
                    self.features = backbone.features
                else:
                    raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")

                self.seg_head = nn.Sequential(
                    nn.Conv2d(out_ch, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.SiLU(inplace=True),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(256, 1, 1),
                )
                self.cls_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(out_ch, 256),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                )

            def forward(self, x):
                input_hw = x.shape[-2:]
                feat = _ensure_nchw(self.features(x))
                seg = self.seg_head(feat)
                seg = F.interpolate(seg, size=input_hw, mode="bilinear", align_corners=False)
                cls = self.cls_head(feat).squeeze(1)
                return {"seg_logits": seg, "cls_logits": cls}

        return _Model()


class SegFormerLite:
    def __new__(cls, in_channels: int = 3, base_channels: int = 48, dropout: float = 0.1):
        torch, nn, F = _torch()

        class MixBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
                self.mix = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )

            def forward(self, x):
                y = self.proj(x)
                return y + self.mix(y)

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
                self.s1 = MixBlock(in_channels, c1, 2)
                self.s2 = MixBlock(c1, c2, 2)
                self.s3 = MixBlock(c2, c3, 2)
                self.s4 = MixBlock(c3, c4, 2)
                self.fuse = nn.Sequential(
                    nn.Conv2d(c1 + c2 + c3 + c4, 256, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
                self.seg_head = nn.Conv2d(256, 1, 1)
                self.cls_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c4, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                )

            def forward(self, x):
                input_hw = x.shape[-2:]
                f1 = self.s1(x)
                f2 = self.s2(f1)
                f3 = self.s3(f2)
                f4 = self.s4(f3)
                target = f1.shape[-2:]
                fused = self.fuse(
                    torch.cat(
                        [
                            f1,
                            F.interpolate(f2, size=target, mode="bilinear", align_corners=False),
                            F.interpolate(f3, size=target, mode="bilinear", align_corners=False),
                            F.interpolate(f4, size=target, mode="bilinear", align_corners=False),
                        ],
                        dim=1,
                    )
                )
                seg = F.interpolate(self.seg_head(fused), size=input_hw, mode="bilinear", align_corners=False)
                cls = self.cls_head(f4).squeeze(1)
                return {"seg_logits": seg, "cls_logits": cls}

        return _Model()


class HFSegFormerB0:
    def __new__(cls, pretrained: bool = False, dropout: float = 0.1):
        torch, nn, F = _torch()
        try:
            from transformers import SegformerConfig, SegformerModel
        except Exception as exc:
            raise ImportError("Install transformers to use hf_segformer_b0") from exc

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if pretrained:
                    self.encoder = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
                    hidden = list(self.encoder.config.hidden_sizes)
                else:
                    cfg = SegformerConfig(num_channels=3, depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256])
                    self.encoder = SegformerModel(cfg)
                    hidden = list(cfg.hidden_sizes)
                self.proj = nn.ModuleList([nn.Conv2d(ch, 128, 1) for ch in hidden])
                self.fuse = nn.Sequential(
                    nn.Conv2d(128 * len(hidden), 256, 1),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
                self.seg_head = nn.Conv2d(256, 1, 1)
                self.cls_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(hidden[-1], 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                )

            def forward(self, x):
                input_hw = x.shape[-2:]
                out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
                feats = [_ensure_nchw_channels(f, ch) for f, ch in zip(out.hidden_states, [p.in_channels for p in self.proj])]
                target = feats[0].shape[-2:]
                fused = []
                for feat, proj in zip(feats, self.proj):
                    fused.append(F.interpolate(proj(feat), size=target, mode="bilinear", align_corners=False))
                fused = self.fuse(torch.cat(fused, dim=1))
                seg = F.interpolate(self.seg_head(fused), size=input_hw, mode="bilinear", align_corners=False)
                cls = self.cls_head(feats[-1]).squeeze(1)
                return {"seg_logits": seg, "cls_logits": cls}

        return _Model()


def create_advanced_model(cfg: dict[str, Any]):
    model_cfg = cfg.get("model", {})
    arch = str(model_cfg.get("architecture", "")).lower()
    pretrained = bool(model_cfg.get("pretrained", False))
    dropout = float(model_cfg.get("dropout", 0.1))
    base = int(model_cfg.get("base_channels", 48))
    if arch in {"convnext_tiny", "convnext_small", "swin_t", "swin_s"}:
        return TorchvisionBackboneSeg(arch, dropout=dropout, pretrained=pretrained)
    if arch == "segformer_lite":
        return SegFormerLite(base_channels=base, dropout=dropout)
    if arch in {"hf_segformer_b0", "segformer_b0"}:
        return HFSegFormerB0(pretrained=pretrained, dropout=dropout)
    raise ValueError(f"Unknown advanced architecture: {arch}")
