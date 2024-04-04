import torch
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
import numpy as np

from .. import config as cfg


def generate_single_explanation(model, exp, input, output):
    _, output = output.max(1)
    model = model.to(cfg.device)
    input = input.to(cfg.device)
    output = output.to(cfg.device)

    if 'lrp' in exp:
        layers = list(model._modules["encoder"]._modules["main"]) + list(model._modules["decoder"]._modules["main"])
        num_layers = len(layers)
        rule = exp.split("_")[1]
        for idx_layer in range(1, num_layers):
            if rule == "gamma":
                setattr(layers[idx_layer], "rule", GammaRule())
            elif rule == "epsilon":
                setattr(layers[idx_layer], "rule", EpsilonRule())
            elif rule == "alpha1beta0":
                setattr(layers[idx_layer], "rule", Alpha1_Beta0_Rule())
        lrp = LRP(model)
        attr = lrp.attribute(input, target=output)
    elif 'ig' in exp:
        ig = IntegratedGradients(model)
        attr = torch.zeros_like(input)
        for i in range(input.shape[0] // 16):
            attr[16 * i:16 * (i + 1)] = ig.attribute(input[16 * i:16 * (i + 1)], target=output[16 * i:16 * (i + 1)])
            if i == (input.shape[0] // 16) - 1 and input.shape[0] % 16 != 0:
                attr[16 * (i + 1):] = ig.attribute(input[16 * (i + 1):], target=output[16 * (i + 1):])
    elif 'occlusion' in exp:
        occlusion = Occlusion(model)
        size = int(exp.split("_")[1])
        attr = occlusion.attribute(input,
                                   strides=(1, size, size),
                                   target=output,
                                   sliding_window_shapes=(1, size, size),
                                   baselines=0)
    elif 'shap' in exp:
        gradient_shap = GradientShap(model)

        attr = torch.zeros_like(input)
        for i in range(input.shape[0] // 16):
            torch.manual_seed(0)
            np.random.seed(0)
            rand_img_dist = torch.cat([input[16 * i:16 * (i + 1)] * 0, input[16 * i:16 * (i + 1)] * 1])
            attr[16 * i:16 * (i + 1)] = gradient_shap.attribute(input[16 * i:16 * (i + 1)],
                                                                n_samples=16,
                                                                stdevs=0.0001,
                                                                baselines=rand_img_dist[16 * i:16 * (i + 1)],
                                                                target=output[16 * i:16 * (i + 1)])
            if i == (input.shape[0] // 16) - 1 and input.shape[0] % 16 != 0:
                attr[16 * (i + 1):] = gradient_shap.attribute(input[16 * (i + 1):],
                                                              n_samples=16,
                                                              stdevs=0.0001,
                                                              baselines=rand_img_dist[16 * (i + 1):],
                                                              target=output[16 * (i + 1):])

    attr /= torch.amax(attr, (2, 3))[:, :, None, None]
    return torch.split(attr, cfg.time_steps if not cfg.mean_input else 1, dim=1)


def generate_explanations(model, input, output):
    torch.cuda.empty_cache()
    input.requires_grad_(True)

    all_explanations = []
    for exp in cfg.explanation_names:
        explanation = generate_single_explanation(model, exp, input, output)
        all_explanations.append(torch.stack(explanation, dim=1))
    all_explanations = torch.stack(all_explanations)
    return all_explanations
