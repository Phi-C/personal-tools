#! /env/bin/python3

from typing import Any, List
import torch.nn as nn


def shape_infer_hook(name):
    """A closure that captures the module name"""

    def hook(module, input, output):
        if len(list(module.children())) == 0:
            print(f"Shape of output to {name} is {output.shape}.")

    return hook


def reason_shape(net: nn.Module, inp: Any, name_list: List[str] = None):
    """Register forward hooks for all layers in a network."""
    for name, layer in net.named_modules():
        if name_list is None or name in name_list:
            layer.register_forward_hook(shape_infer_hook(name))

    net(inp)


if __name__ == "__main__":
    import torch
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    model.eval()
    inp = torch.randn(3, 3, 224, 224)
    reason_shape(model, inp)
