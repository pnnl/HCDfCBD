"""
A non-variational version of Deep-IMV.  We just average the predictions of each marginal model instead of computing some joint distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class simple_FC(nn.Module):
    """
    A simple MLP for a single view of the multi-view model.

    Attributes:
        input_size (int): The number of input features
        hidden_sizes (List[int]): The number of hidden units in each layer
        prediction_dim (int): The number of output classes
        dropout (float, optional): The dropout rate. Defaults to 0.2.
        fc1 (nn.Linear): The first fully connected layer after the input
        fc{j} (nn.Linear): The j-th fully connected layer after the input layer
        fc_out (nn.Linear): The layer that maps the (sampled) latent representation to the output classes
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], prediction_dim: int, dropout: float = 0.2):
        """
        Initialize the FC_Marginal model

        Args:
            input_size (int): The number of input features
            hidden_sizes (List[int]): The number of hidden units in each layer
            prediction_dim (int): The number of output classes
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.prediction_dim = prediction_dim
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        for sz in range(1, len(hidden_sizes)):
            setattr(self, f'fc{sz+1}', nn.Linear(hidden_sizes[sz-1], hidden_sizes[sz]))

        self.fc_out = nn.Linear(hidden_sizes[-1], prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = F.relu(self.fc1(x))

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(F.relu(getattr(self, f'fc{sz+1}')(h)))
        
        preds = self.dropout(self.fc_out(h))
        preds = F.softmax(preds, dim=-1)
        
        return preds, h
    
class JointMLP(nn.Module):
    """
    A model that fuses the predictions of multiple marginal models.

    Attributes:
        margin_models (torch.nn.ModuleList): A list of marginal models of type simple_FC
        fc1 (nn.Linear): The first fully connected layer after the last hidden layer of the marginal models
        fc2 (nn.Linear): The second fully connected layer, immediately after fc1
        dropout (nn.Dropout): A dropout layer
    """
    def __init__(self, marginal_models: List[simple_FC], hidden_dim: int = 128, dropout: float = 0.2):
        """
        Initialize the JointMLP model

        Args:
            marginal_models (List[simple_FC]): A list of marginal models of type simple_FC
            hidden_dim (int, optional): The number of hidden units between fc1 and fc2. Defaults to 128.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.margin_models = torch.nn.ModuleList(marginal_models)
        assert len(set([m.hidden_sizes[-1] for m in self.margin_models])) == 1, "All models must have the same last hidden size"
        self.fc1 = nn.Linear(marginal_models[0].hidden_sizes[-1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.margin_models[0].prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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
    
    def loss(self, y, yhat, yhats, focal=True, gamma=2., alpha=None, **kwargs) -> torch.Tensor:
        """ Compute the total loss for all the joint and marginal models

        Args:
            y (torch.Tensor): Ground truth labels
            yhat (torch.Tensor): softmax predictions for the combinations of experts
            yhats (List[torch.Tensor]): List of softmax predictions for each marginal model

        Returns:
            torch.Tensor: The joint loss
        """

        product_loss = F.cross_entropy(yhat, y, reduction='none')
        marginal_losses = [F.cross_entropy(yh, y, reduction='none') for yh in yhats]

        if alpha is not None:
            alpha = alpha.repeat(yhat.shape[0], 1).to(yhat.device)
            alpha = alpha.gather(1, y.view(-1, 1))
            product_loss = product_loss * alpha.view(-1)
            marginal_losses = [m * alpha.view(-1) for m in marginal_losses]

        if focal:
            product_loss = torch.pow(1 - yhat.gather(1, y.view(-1, 1)), gamma).view(-1) * product_loss
            marginal_losses = [torch.pow(1 - yh.gather(1, y.view(-1, 1)), gamma).view(-1) * m for yh, m in zip(yhats, marginal_losses)]
        
        # import pdb; pdb.set_trace()
        product_loss = torch.mean(product_loss)
        marginal_losses = [torch.mean(m) for m in marginal_losses]

        return product_loss + sum(marginal_losses)
    
