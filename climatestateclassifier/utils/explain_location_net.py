import torch

from .. import config as cfg


def generate_single_explanation(model, rule, input):
    # Forward pass
    y_hat_lrp = model.forward(input.to(cfg.device), explain=True, rule=rule, pattern=None)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(input.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()

    # Backward pass (compute explanation)
    y_hat_lrp.backward()
    attr = input.grad
    return torch.split(attr, cfg.time_steps if not cfg.mean_input else 1, dim=1)


def generate_explanations(model, input):
    torch.cuda.empty_cache()
    input.requires_grad_(True)

    all_explanations = []
    for exp in cfg.explanation_names:
        explanation = generate_single_explanation(model, exp, input)
        all_explanations.append(torch.stack(explanation, dim=1))
    all_explanations = torch.stack(all_explanations)
    return all_explanations


