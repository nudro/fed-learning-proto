"""
Model definition for ResNet18 transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Create ResNet18 model for transfer learning
    
    Args:
        num_classes: Number of output classes (default: 2 for ants/bees)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze the backbone layers
    
    Returns:
        PyTorch model
    """
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model


def get_model_parameters(model):
    """Get model parameters as list of numpy arrays for Flower"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    """Set model parameters from list of numpy arrays"""
    model_keys = list(model.state_dict().keys())
    
    # Validate parameter shapes match
    if len(parameters) != len(model_keys):
        raise ValueError(
            f"Parameter count mismatch: model has {len(model_keys)} layers, "
            f"received {len(parameters)} parameters"
        )
    
    # Check shapes match
    for i, (key, param) in enumerate(zip(model_keys, parameters)):
        model_param = model.state_dict()[key]
        if model_param.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for layer '{key}': "
                f"model expects {model_param.shape}, received {param.shape}"
            )
    
    params_dict = zip(model_keys, parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model

