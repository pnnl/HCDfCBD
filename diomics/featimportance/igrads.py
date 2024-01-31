import torch
from typing import List

def integrated_grads(views, baselines, model, n_steps = 100, class_idx = 1):
    """Compute integrated gradients for a given model and input.

    Args:
        views List[torch.tensor]: A list of tensors for the multiview model.
        baselines List[torch.tensor]: List of baseline inputs, with dimensions the same as the views.
        model (torch.nn.Module): The model with which to perform gradient analysis that accepts inputs like views and baselines.
        n_steps (int, optional): Number of points along the path from the baseline to the input on which to compute the gradients. Defaults to 100.
        class_idx (int, optional): The index of the output corresponding to the positive class. Defaults to 1.

    Returns:
        List[torch.tensor]: The integrated gradients for each view/sample.
    """
    grads = [torch.zeros_like(v) for v in baselines]
    for i in range(1, n_steps + 1):
        tmp = [torch.clone(b) for b in baselines]
        tmp = [b + (i/n_steps) * (vtest - b) for b, vtest in zip(tmp, views)]
        tmp = [b.requires_grad_() for b in tmp]
        yhat, poe_dist, yhats, dists = model(tmp)
        yhat[:,class_idx].mean().backward()
        grads = [g + b.grad for g, b in zip(grads, tmp)]

    return [g * (v - b) / n_steps for g, b, v in zip(grads, baselines, views)]