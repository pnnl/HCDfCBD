"""
A non-variational version of Deep-IMV.  We just average the predictions of each marginal model instead of computing some joint distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Any, cast

class simple_FC(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], prediction_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.prediction_dim = prediction_dim
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        for sz in range(1, len(hidden_sizes)):
            setattr(self, f'fc{sz+1}', nn.Linear(hidden_sizes[sz-1], hidden_sizes[sz]))

        self.fc_out = nn.Linear(hidden_sizes[-1], prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = F.relu(self.fc1(x))

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(F.relu(getattr(self, f'fc{sz+1}')(h)))
        
        preds = self.dropout(self.fc_out(h))
        preds = F.softmax(preds, dim=-1)
        
        return preds, h
    
class JointMLP(nn.Module):
    def __init__(self, marginal_models: List[simple_FC], hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.margin_models = torch.nn.ModuleList(marginal_models)
        assert len(set([m.hidden_sizes[-1] for m in self.margin_models])) == 1, "All models must have the same last hidden size"
        self.fc1 = nn.Linear(marginal_models[0].hidden_sizes[-1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.margin_models[0].prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: List[torch.Tensor]):
        assert len(x) == len(self.margin_models), "Number of inputs must match number of marginal models"

        yhats = []
        hiddens = []

        # maintain a queue of poe_dists
        # separate the input tensors into batches with 1, 2, 3, ... complete views
        # for each batch, compute the distributions and fine the poe_dist for that batch, append it to the queue of poe_dists
        # once you've gone through all batches
        # Nono, do the batching thing outside this loop, but accumulate the losses (add em up), then call backwards() on the sum

        for i, model in enumerate(self.margin_models):
            # ignore view-missing data
            if not isinstance(x[i], torch.Tensor):
                continue

            yhat, h = model(x[i])
            yhats.append(yhat)
            hiddens.append(h)

        h = torch.mean(torch.stack(hiddens), dim=0)

        h = self.dropout(F.relu(self.fc1(h)))
        yhat = F.softmax(self.fc2(h), dim=-1)

        return yhat, h, yhats, hiddens
    
    def loss(self, y, yhat, yhats, alpha = 0.5, **kwargs):
        """ Compute the total loss for all the joint and marginal models

        Args:
            y (torch.Tensor): Ground truth labels
            yhat (torch.Tensor): softmax predictions for the combinations of experts
            yhats (List[torch.Tensor]): List of softmax predictions for each marginal model

        Returns:
            torch.Tensor: The joint loss
        """

        product_loss = F.cross_entropy(yhat, y)

        # if alpha is not None:
        #     alpha = alpha.repeat(yhat.shape[0], 1).to(yhat.device)
        #     alpha = alpha.gather(1, y.view(-1, 1))
        #     ce = ce * alpha.view(-1)

        # if focal:
        #     focal_loss = torch.pow(1 - yhat.gather(1, y.view(-1, 1)), gamma).view(-1) * ce
        #     focal_loss = focal_loss.sum()
        #     return torch.mean(focal_loss + var_beta * kl)
        # else:
        #     ce = ce.sum()
        #     return torch.mean(ce + var_beta * kl)  

        marginal_losses = [F.cross_entropy(yh, y) for yh in yhats]
    
        return alpha * product_loss + (1-alpha) * sum(marginal_losses)
