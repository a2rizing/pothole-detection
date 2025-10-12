"""
Custom lightweight CNN architectures for pothole detection
Optimized for Raspberry Pi and real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, efficientnet_b0
import math
from typing import Optional, Tuple

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for reducing parameters and computation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetV2 style)
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Pointwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PotholeNet(nn.Module):
    """
    Custom lightweight CNN for pothole detection and classification
    Supports both classification and optional depth estimation
    """
    
    def __init__(self, 
                 num_classes=2, 
                 input_size=(224, 224),
                 with_depth=False,
                 dropout_rate=0.3,
                 width_multiplier=1.0):
        """
        Args:
            num_classes (int): Number of output classes
            input_size (tuple): Input image size (H, W)
            with_depth (bool): Whether to include depth estimation branch
            dropout_rate (float): Dropout rate
            width_multiplier (float): Width multiplier for channels
        """
        super(PotholeNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.with_depth = with_depth
        
        # Calculate channel dimensions
        def make_divisible(v, divisor=8):
            return math.ceil(v / divisor) * divisor
        
        # Feature extraction backbone
        channels = [
            make_divisible(32 * width_multiplier),
            make_divisible(64 * width_multiplier),
            make_divisible(128 * width_multiplier),
            make_divisible(256 * width_multiplier),
            make_divisible(512 * width_multiplier)
        ]
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks
        self.features = nn.Sequential(
            # Block 1
            InvertedResidual(channels[0], channels[1], stride=2, expand_ratio=1),
            InvertedResidual(channels[1], channels[1], stride=1, expand_ratio=6),
            
            # Block 2
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=6),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=6),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=6),
            
            # Block 3
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=6),
            InvertedResidual(channels[3], channels[3], stride=1, expand_ratio=6),
            InvertedResidual(channels[3], channels[3], stride=1, expand_ratio=6),
            SEBlock(channels[3]),
            
            # Block 4
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=6),
            InvertedResidual(channels[4], channels[4], stride=1, expand_ratio=6),
            SEBlock(channels[4])
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(channels[4], channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[3], num_classes)
        )
        
        # Severity regression head (for pothole severity estimation)
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(channels[4], channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[3], 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Optional depth estimation branch
        if with_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(channels[4], channels[3], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[3], channels[2], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[0], 1, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        features = self.features(x)
        
        # Global average pooling for classification
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        
        # Classification output
        class_output = self.classifier(pooled)
        
        # Severity output
        severity_output = self.severity_head(pooled)
        
        outputs = {
            'classification': class_output,
            'severity': severity_output
        }
        
        # Optional depth estimation
        if self.with_depth:
            depth_output = self.depth_head(features)
            # Resize to match input size
            depth_output = F.interpolate(depth_output, size=self.input_size, 
                                       mode='bilinear', align_corners=False)
            outputs['depth'] = depth_output
        
        return outputs
    
    def get_feature_maps(self, x):
        """Get intermediate feature maps for visualization"""
        x = self.conv1(x)
        feature_maps = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [1, 4, 9, 12]:  # Save specific feature maps
                feature_maps.append(x)
        
        return feature_maps


class MiniPotholeNet(nn.Module):
    """
    Ultra-lightweight version for very constrained devices
    """
    
    def __init__(self, num_classes=2, input_size=(224, 224)):
        super(MiniPotholeNet, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Extremely lightweight backbone
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            # Depthwise blocks
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            
            # Final feature extraction
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Severity head
        self.severity_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        return {
            'classification': self.classifier(features),
            'severity': self.severity_head(features)
        }


class PotholeDetector(nn.Module):
    """
    Object detection version using anchor-free approach
    Suitable for real-time pothole detection
    """
    
    def __init__(self, num_classes=2, input_size=(416, 416), backbone='mobilenet'):
        super(PotholeDetector, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone selection
        if backbone == 'mobilenet':
            self.backbone = self._build_mobilenet_backbone()
            backbone_channels = 960
        elif backbone == 'custom':
            self.backbone = self._build_custom_backbone()
            backbone_channels = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Feature Pyramid Network (FPN) for multi-scale detection
        self.fpn = self._build_fpn(backbone_channels)
        
        # Detection heads
        self.cls_head = self._build_detection_head(256, num_classes)
        self.reg_head = self._build_detection_head(256, 4)  # x, y, w, h
        self.obj_head = self._build_detection_head(256, 1)  # objectness
        
        # Severity head for detected objects
        self.severity_head = self._build_detection_head(256, 1)
    
    def _build_mobilenet_backbone(self):
        """Build MobileNetV3 backbone"""
        mobilenet = mobilenet_v3_small(pretrained=True)
        # Remove classifier and avgpool
        features = list(mobilenet.features.children())
        return nn.Sequential(*features)
    
    def _build_custom_backbone(self):
        """Build custom lightweight backbone"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            InvertedResidual(32, 64, stride=2),
            InvertedResidual(64, 128, stride=2),
            InvertedResidual(128, 256, stride=2),
            InvertedResidual(256, 512, stride=2),
            
            SEBlock(512)
        )
    
    def _build_fpn(self, backbone_channels):
        """Build Feature Pyramid Network"""
        return nn.Sequential(
            nn.Conv2d(backbone_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def _build_detection_head(self, in_channels, out_channels):
        """Build detection head"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1)
        )
    
    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        
        # FPN processing
        fpn_features = self.fpn(features)
        
        # Detection heads
        cls_output = self.cls_head(fpn_features)
        reg_output = self.reg_head(fpn_features)
        obj_output = self.obj_head(fpn_features)
        severity_output = self.severity_head(fpn_features)
        
        return {
            'classification': cls_output,
            'regression': reg_output,
            'objectness': obj_output,
            'severity': severity_output
        }


def create_model(model_type='potholeNet', **kwargs):
    """
    Factory function to create different model variants
    
    Args:
        model_type (str): Type of model to create
        **kwargs: Additional arguments for model creation
    
    Returns:
        torch.nn.Module: Created model
    """
    
    if model_type.lower() == 'potholenet':
        return PotholeNet(**kwargs)
    elif model_type.lower() == 'mini':
        return MiniPotholeNet(**kwargs)
    elif model_type.lower() == 'detector':
        return PotholeDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(3, 224, 224)):
    """Print model summary"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, *input_size)
    with torch.no_grad():
        output = model(dummy_input)
    
    if isinstance(output, dict):
        for key, value in output.items():
            print(f"Output '{key}': {value.shape}")
    else:
        print(f"Output shape: {output.shape}")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing PotholeNet models...")
    
    # Test standard PotholeNet
    print("\n1. Testing PotholeNet:")
    model = create_model('potholeNet', num_classes=2, with_depth=True)
    model_summary(model, input_size=(3, 224, 224))
    
    # Test MiniPotholeNet
    print("\n2. Testing MiniPotholeNet:")
    mini_model = create_model('mini', num_classes=2)
    model_summary(mini_model, input_size=(3, 224, 224))
    
    # Test PotholeDetector
    print("\n3. Testing PotholeDetector:")
    detector = create_model('detector', num_classes=2, input_size=(416, 416))
    model_summary(detector, input_size=(3, 416, 416))
    
    # Compare model sizes
    print("\nModel comparison:")
    models = {
        'PotholeNet': model,
        'MiniPotholeNet': mini_model,
        'PotholeDetector': detector
    }
    
    for name, m in models.items():
        params = count_parameters(m)
        print(f"{name}: {params:,} parameters ({params/1e6:.2f}M)")
    
    print("\nAll models created successfully!")