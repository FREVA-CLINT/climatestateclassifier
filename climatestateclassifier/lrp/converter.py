import torch
from .conv import Conv2d
from .linear import Linear
from .sequential import Sequential
from climatestateclassifier.model.decoder import LocationDecoder
from climatestateclassifier.model.encoder import Encoder, EncoderBlock

conversion_table = {
    'Linear': Linear,
    'Conv2d': Conv2d
}


def convert_location_net(module, modules=None):
    # First time
    if modules is None:
        modules = []
        for m in module.children():
            convert_location_net(m, modules=modules)

            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, LocationDecoder) or isinstance(module, Encoder):
        convert_location_net(module.main, modules=modules)
    elif isinstance(module, EncoderBlock):
        convert_location_net(module.conv, modules=modules)
        if hasattr(module, "pooling"):
            convert_location_net(module.pooling, modules=modules)
        if hasattr(module, "bn"):
            convert_location_net(module.bn, modules=modules)
        if hasattr(module, "activation"):
            convert_location_net(module.activation, modules=modules)
    elif isinstance(module, torch.nn.Sequential):
        for m in module.children():
            convert_location_net(m, modules=modules)
    elif isinstance(module, torch.nn.ModuleList):
        for m in module:
            convert_location_net(m, modules=modules)
    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU):
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    else:
        modules.append(module)
