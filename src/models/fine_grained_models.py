"""Advanced models for fine-grained visual categorization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict, Any
import math


class ChannelAttention(nn.Module):
    """Channel Attention Module."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """Efficient Channel Attention Block."""
    
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class FineGrainedResNet(nn.Module):
    """ResNet with attention mechanisms for fine-grained categorization."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 200,
        attention_type: str = "cbam",
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add attention mechanism
        if attention_type == "cbam":
            self.attention = CBAM(feature_dim)
        elif attention_type == "se":
            self.attention = SEBlock(feature_dim)
        elif attention_type == "eca":
            self.attention = ECABlock(feature_dim)
        else:
            self.attention = None
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        if self.attention is not None:
            features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class FineGrainedViT(nn.Module):
    """Vision Transformer for fine-grained categorization."""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 200,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        
        try:
            import timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            feature_dim = self.backbone.num_features
        except ImportError:
            raise ImportError("timm is required for Vision Transformer models")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for fine-grained categorization."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 200,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classifier and get intermediate layers
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6:7])
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7:8])
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, 1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize weights
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract multi-scale features
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Fuse features
        fused = self.fusion(x4)
        
        # Global pooling
        pooled = self.global_pool(fused)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


def create_model(
    model_name: str,
    num_classes: int = 200,
    **kwargs
) -> nn.Module:
    """Create a model instance."""
    
    if model_name.startswith("resnet"):
        return FineGrainedResNet(
            backbone=model_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name.startswith("vit"):
        return FineGrainedViT(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_name == "multiscale_resnet50":
        return MultiScaleFeatureExtractor(
            backbone="resnet50",
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
