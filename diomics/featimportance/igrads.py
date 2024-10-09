import torch
from typing import List

def integrated_grads(views, baselines, model, n_steps = 100, class_idx = 1, activation_layer = None, layer_depth = 'marginal'):
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

    afunc = torch.nn.functional.relu

    if activation_layer is None:
        grads = [torch.zeros_like(v) for v in baselines]
    else:
        yhat, poe_dist, yhats, dists = model(*views)
        yhat[:,class_idx].mean().backward()

        if layer_depth == 'marginal':
            # get the activations for original input

            activations = [afunc(m.activations[activation_layer]) for m in model.margin_models]

            tmp = [torch.clone(b) for b in baselines]
            tmp = [b.requires_grad_() for b in tmp]

            # get baseline activations and gradients
            yhat, poe_dist, yhats, dists = model(*tmp)
            yhat[:,class_idx].mean().backward(retain_graph = True)

            baseline_activations = [afunc(m.activations[activation_layer]) for m in model.margin_models]
            grads = [m.activations_grad[activation_layer] for m in model.margin_models]

            input_grads = []

            for i,m in enumerate(model.margin_models):
                input_grads_stack = []
                for j in range(m.activations[activation_layer].size(-1)):
                    afunc(m.activations[activation_layer]).mean(axis=0)[j].backward(retain_graph = True)
                    input_grads_stack.append(tmp[i].grad.clone())

                input_grads.append(torch.stack(input_grads_stack, axis = 0))
        else:
            activations = afunc(model.activations[activation_layer])

            tmp = [torch.clone(b) for b in baselines]
            tmp = [b.requires_grad_() for b in tmp]

            # get baseline activations and gradients
            yhat, poe_dist, yhats, dists = model(*tmp)
            yhat[:,class_idx].mean().backward(retain_graph = True)

            baseline_activations = afunc(model.activations[activation_layer])
            grads = model.activations_grad[activation_layer]

            input_grads = []

            for i in range(len(tmp)):
                input_grads_stack = []
                for j in range(model.activations[activation_layer].size(-1)):
                    afunc(model.activations[activation_layer]).mean(axis=0)[j].backward(retain_graph = True)
                    input_grads_stack.append(tmp[i].grad.clone())

                input_grads.append(torch.stack(input_grads_stack, axis = 0))

    if activation_layer is None:
        for i in range(1, n_steps + 1):
            tmp = [torch.clone(b) for b in baselines]
            # here convert tmp to the activations at each layer for the baselines

            tmp = [b + (i/n_steps) * (vtest - b) for b, vtest in zip(tmp, views)]
            tmp = [b.requires_grad_() for b in tmp]
            yhat, poe_dist, yhats, dists = model(*tmp)
            yhat[:,class_idx].mean().backward()
            grads = [g + b.grad for g, b in zip(grads, tmp)]

        return [g * (v - b) / n_steps for g, b, v in zip(grads, baselines, views)]
    else:
        if layer_depth == 'marginal':
            for i in range(2, n_steps + 1):
                tmp = [torch.clone(b) for b in baselines]
                tmp = [b + (i/n_steps) * (vtest - b) for b, vtest in zip(tmp, views)]
                tmp = [b.requires_grad_() for b in tmp]

                yhat, poe_dist, yhats, dists = model(*tmp)
                yhat[:,class_idx].mean().backward(retain_graph=True)

                grads = [g + m.activations_grad[activation_layer] for g, m in zip(grads, model.margin_models)]

                # now we need an inner loop that gets the integrated gradients for the input with respect to each activation in the intermediate layer.
                
                for j,m in enumerate(model.margin_models):
                    input_grads_stack = []
                    for k in range(m.activations[activation_layer].size(-1)):
                        afunc(m.activations[activation_layer]).mean(axis=0)[k].backward(retain_graph = True)
                        input_grads_stack.append(tmp[j].grad.clone())

                    input_grads[j] += torch.stack(input_grads_stack, axis = 0)

            input_igrads = [g * (v - b) / n_steps for g, b, v in zip(input_grads, baselines, views)]     
            activation_igrads = [g * (a - b) / n_steps for g, b, a in zip(grads, baseline_activations, activations)]
        else:
            for i in range(2, n_steps + 1):
                tmp = [torch.clone(b) for b in baselines]
                tmp = [b + (i/n_steps) * (vtest - b) for b, vtest in zip(tmp, views)]
                tmp = [b.requires_grad_() for b in tmp]

                yhat, poe_dist, yhats, dists = model(*tmp)
                yhat[:,class_idx].mean().backward(retain_graph=True)

                grads = grads + model.activations_grad[activation_layer]

                # now we need an inner loop that gets the integrated gradients for the input with respect to each activation in the intermediate layer.
                
                for j in range(len(tmp)):
                    input_grads_stack = []
                    for k in range(model.activations[activation_layer].size(-1)):
                        afunc(model.activations[activation_layer]).mean(axis=0)[k].backward(retain_graph = True)
                        input_grads_stack.append(tmp[j].grad.clone())

                input_grads.append(torch.stack(input_grads_stack, axis = 0))

            input_igrads = [g * (v - b) / n_steps for g, b, v in zip(input_grads, baselines, views)]     
            activation_igrads = grads * (activations - baseline_activations) / n_steps

        return {
            "input_igrads": input_igrads,
            "activation_igrads": activation_igrads
        }

# need to get hooks to get activation of the target layer for each view as well as each baseline
# then extract the gradients from that layer and do the usual.

def average_igrads(views, model, cat_names, baselines = None, n_runs = 20, n_steps = 100, **kwargs):
    all_scores = [{} for _ in range(len(views))]

    for i in range(n_runs):
        for class_idx, name in enumerate(cat_names):
            # should make this high signal-to-noise
            if baselines is None:
                baselines = [t + torch.randn_like(t)*t.std()*2 for t in views]
            else:
                baselines = [b + torch.randn_like(b)*b.std()*2 for b in baselines]

            igrads_out = integrated_grads(
                views, 
                baselines, 
                model, 
                n_steps = n_steps, 
                class_idx = class_idx,
                **kwargs
            )

            for j, res_dict in enumerate(all_scores):
                res_dict.setdefault(name, []).append(igrads_out[j])

    return all_scores