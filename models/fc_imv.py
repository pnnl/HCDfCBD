import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from typing import List, Tuple, Dict, Union, Any, cast

class FC_Marginal(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], z_dim: int, prediction_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.z_dim = z_dim
        self.prediction_dim = prediction_dim
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        for sz in range(1, len(hidden_sizes)):
            setattr(self, f'fc{sz+1}', nn.Linear(hidden_sizes[sz-1], hidden_sizes[sz]))

        self.fc_z = nn.Linear(hidden_sizes[-1], z_dim * 2)
        self.fc_out = nn.Linear(z_dim, prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = F.relu(self.fc1(x))

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(F.relu(getattr(self, f'fc{sz+1}')(h)))

        h = self.fc_z(h)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        
        dist = Normal(mu, torch.exp(logvar / 2))

        z = dist.rsample()
        out = self.dropout(self.fc_out(z))
        out = F.softmax(out, dim=-1)
        
        return out, dist

def product_of_experts(dists):
    """ Compute the mean and variance of the product of N normal distributions each having a N(mu_i, S_i) distribution with a N(mu_0, S_0) prior.  The covariance matrix and mean of the product of experts is given by:

    cov_poe = (1/S_0 + sum_i (1/S_i) )^-1
    mu_poe = (mu_0 * 1/S_0 + sum_i (mu_i * 1/S_i)^-1) * cov_poe

    Args:
        dists (List[Normal]): List of torch.distributions.Normal

    Returns:
        torch.distributions.Normal: The product of experts distribution
    """
    # assume mu_0 = 0, S_0 = I
    var_poe = 1
    mu_poe = 0

    for dist in dists:
        assert isinstance(dist, Normal), "Only Normal distributions are supported"
        var_ = dist.variance
        mu_ = dist.mean

        var_poe = var_poe + torch.div(1., var_)
        mu_poe = mu_poe + torch.div(mu_, var_)

    var_poe = torch.div(1., var_poe)
    mu_poe = torch.mul(mu_poe, var_poe)

    return Normal(mu_poe, var_poe)

class JointVAE(nn.Module):
    def __init__(self, marginal_models: List[FC_Marginal], hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.margin_models = torch.nn.ModuleList(marginal_models)
        self.fc1 = nn.Linear(self.margin_models[0].z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.margin_models[0].prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: List[torch.Tensor]):
        assert len(x) == len(self.margin_models), "Number of inputs must match number of marginal models"

        dists = []
        yhats = []

        # maintain a queue of poe_dists
        # separate the input tensors into batches with 1, 2, 3, ... complete views
        # for each batch, compute the distributions and fine the poe_dist for that batch, append it to the queue of poe_dists
        # once you've gone through all batches
        # Nono, do the batching thing outside this loop, but accumulate the losses (add em up), then call backwards() on the sum

        for i, model in enumerate(self.margin_models):
            # ignore view-missing data
            if not isinstance(x[i], torch.Tensor):
                continue

            yhat, dist = model(x[i])

            dists.append(dist)
            yhats.append(yhat)

        # compute the mean and variance of the product of normal distributions
        poe_dist = product_of_experts(dists)

        if self.training:
            z = poe_dist.rsample()
        else:
            z = poe_dist.loc

        h = self.dropout(F.relu(self.fc1(z)))
        yhat = F.softmax(self.fc2(h), dim=-1)

        return yhat, poe_dist, yhats, dists
    
    def loss(self, y, yhat, poe_dist, yhats, dists, **kwargs):
        """ Compute the total loss for all the joint and marginal models

        Args:
            y (torch.Tensor): Ground truth labels
            yhat (torch.Tensor): softmax predictions for the product of experts
            poe_dist (torch.distributions.Normal): The distribution of the product of experts
            yhats (List[torch.Tensor]): List of softmax predictions for each marginal model
            dists (List[torch.distributions.Normal]): List of distributions for each marginal model
            var_beta (float, optional): The weight of the KL divergence term. Defaults to 1..

        Returns:
            torch.Tensor: The joint loss
        """

        assert isinstance(poe_dist, Normal), "Only Normal distributions are supported"

        product_loss = var_loss(y, yhat, poe_dist, **kwargs)

        # compute the variational loss for each marginal model
        var_losses = [var_loss(y, yhat, dist, **kwargs) for yhat, dist in zip(yhats, dists)]
    
        return product_loss + sum(var_losses)
    
def var_loss(y, yhat, dist, var_beta=1., focal = True, gamma = 2., alpha = None):
    """ Compute the variational loss for a single marginal model

    Args:
        y (torch.Tensor): Ground truth labels
        yhat (torch.Tensor): Prediction probabilities for the labels
        dist (torch.distributions.Normal): The distribution of the latent variable
        var_beta (float, optional): The weight of the KL divergence term. Defaults to 1.

    Returns:
        torch.Tensor: The variational loss
    """

    assert isinstance(dist, Normal), "Only Normal distributions are supported"

    # compute the KL divergence between the prior and the posterior
    kl = kl_divergence(dist, Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.variance))).sum(dim=-1)
    
    # compute the cross entropy loss
    ce = F.cross_entropy(yhat, y, reduction='none')

    if alpha is not None:
        alpha = alpha.repeat(yhat.shape[0], 1).to(yhat.device)
        alpha = alpha.gather(1, y.view(-1, 1))
        ce = ce * alpha.view(-1)

    if focal:
        focal_loss = torch.pow(1 - yhat.gather(1, y.view(-1, 1)), gamma).view(-1) * ce
        focal_loss = focal_loss.sum()
        return torch.mean(focal_loss + var_beta * kl)
    else:
        ce = ce.sum()
        return torch.mean(ce + var_beta * kl)        
