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
    def __init__(self, input_size: int, hidden_sizes: List[int], prediction_dim: int, activation_fn = F.relu, dropout: float = 0.2):
        """
        Initialize the FC_Marginal model

        Args:
            input_size (int): The number of input features
            hidden_sizes (List[int]): The number of hidden units in each layer
            prediction_dim (int): The number of output classes
            activation_fn (torch.nn.functional, optional): The activation function. Defaults to F.relu
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.prediction_dim = prediction_dim
        self.activation_fn = activation_fn
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        for sz in range(1, len(hidden_sizes)):
            setattr(self, f'fc{sz+1}', nn.Linear(hidden_sizes[sz-1], hidden_sizes[sz]))

        self.fc_out = nn.Linear(hidden_sizes[-1], prediction_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = self.activation_fn(self.fc1(x))
        h = self.dropout(h)

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(self.activation_fn(getattr(self, f'fc{sz+1}')(h)))
        
        preds = self.dropout(self.fc_out(h))
        preds = F.softmax(preds, dim=-1)
        
        return preds, h

class simple_FC_hook(simple_FC):
    """
    A simple MLP for a single view of the multi-view model with hooks to retrieve gradient computations.

    Attributes:
        input_size (int): The number of input features
        hidden_sizes (List[int]): The number of hidden units in each layer
        prediction_dim (int): The number of output classes
        dropout (float, optional): The dropout rate. Defaults to 0.2.
        fc1 (nn.Linear): The first fully connected layer after the input
        fc{j} (nn.Linear): The j-th fully connected layer after the input layer
        fc_out (nn.Linear): The layer that maps the (sampled) latent representation to the output classes
        activations (Dict[str, torch.Tensor]): A dictionary of activations for model layers
        activations_grad (Dict[str, torch.Tensor]): A dictionary of gradients for model layers
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the FC_Marginal model

        Args:
            input_size (int): The number of input features
            hidden_sizes (List[int]): The number of hidden units in each layer
            prediction_dim (int): The number of output classes
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__(*args, **kwargs)
        self.fc1.register_forward_hook(self.get_activation('fc1'))
        self.activations = {}
        self.activations_grad = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def get_activation_grad(self, name):
        def hook(grad):
            self.activations_grad[name] = grad
        return hook

    def forward(self, x):
        h = self.dropout(self.activation_fn(self.fc1(x)))

        h.register_hook(self.get_activation_grad('fc1'))

        for sz in range(1, len(self.hidden_sizes)):
            h = self.dropout(self.activation_fn(getattr(self, f'fc{sz+1}')(h)))
        
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
    def __init__(self, marginal_models: List[simple_FC], hidden_dim: int = 128, activation_fn = F.relu, dropout: float = 0.2, combine_fn = "mean", hooks = False):
        """
        Initialize the JointMLP model

        Args:
            marginal_models (List[simple_FC]): A list of marginal models of type simple_FC
            hidden_dim (int, optional): The number of hidden units between fc1 and fc2. Defaults to 128.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.margin_models = torch.nn.ModuleList(marginal_models)

        if combine_fn == 'mean':
            assert len(set([m.hidden_sizes[-1] for m in self.margin_models])) == 1, "If mean combining, all models must have the same last hidden size"
            dim_fc1_in = marginal_models[0].hidden_sizes[-1]
        elif combine_fn == 'concat':
            dim_fc1_in = sum([m.hidden_sizes[-1] for m in self.margin_models])
            
        self.combine_fn = combine_fn
        self.fc1 = nn.Linear(dim_fc1_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.margin_models[0].prediction_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.hooks = hooks

        if self.hooks:
            self.activations = {}
            self.activations_grad = {}

            self.fc1.register_forward_hook(self.get_activation('fc1'))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def get_activation_grad(self, name):
        def hook(grad):
            self.activations_grad[name] = grad
        return hook

    def forward(self, *x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
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

        if self.combine_fn == 'mean':
            h = torch.mean(torch.stack(hiddens), dim=0)
        elif self.combine_fn == 'concat':
            h = torch.cat(hiddens, dim=-1)

        h = self.dropout(self.activation_fn(self.fc1(h)))

        if self.hooks:
            h.register_hook(self.get_activation_grad('fc1'))

        yhat = F.softmax(self.fc2(h), dim=-1)

        return yhat, h, yhats, hiddens
    
    def loss(self, y, yhat, yhats, focal=True, gamma=2., alpha=None, marginal_weight=None, marginal_coefs=None) -> torch.Tensor:
        """ Compute the total loss for all the joint and marginal models

        Args:
            y (torch.Tensor): Ground truth labels
            yhat (torch.Tensor): softmax predictions for the combinations of experts
            yhats (List[torch.Tensor]): List of softmax predictions for each marginal model
            focal (bool, optional): Whether to use focal loss. Defaults to True.
            gamma (float, optional): The focal loss gamma parameter. Defaults to 2.
            alpha (torch.Tensor, optional): A tensor with number of elements equal to the number of classes, specifying class weights. Defaults to None.
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
        
        product_loss = torch.mean(product_loss)
        marginal_losses = [torch.mean(m) for m in marginal_losses]

        if marginal_weight is not None:
            if marginal_coefs is None:
                marginal_coefs = [1.0] * len(marginal_losses)
            marginal_losses = [m * l for m,l in zip(marginal_coefs, marginal_losses)]

            avg_marginal_loss = sum(marginal_losses)/len(marginal_losses)

            loss = product_loss + marginal_weight*avg_marginal_loss
        else:
            loss = product_loss + sum(marginal_losses)/len(marginal_losses) 

        return product_loss, marginal_losses, loss

def make_joint_model(datas, prediction_dim, hidden_sizes, dropout, hidden_dim, activation_fn=F.relu, combine_fn='concat'):
    """
    Create a joint model for multiple views.  Each view gets its own 'marginal model', and then there is a fusion model that takes the output of each marginal model and combines them.

    Args:
        datas (List[torch.Tensor]): A list of tensors, each representing a view
        prediction_dim (int): The number of output classes
        hidden_sizes (List[List[int]]): A list of hidden sizes for each marginal model
        dropout (float): The dropout rate
        hidden_dim (int): The number of hidden units between fc1 and fc2 of the combination model.
        activation_fn (torch.nn.functional): The activation function
        combine_fn (str, optional): The method to combine the marginal models. Defaults to 'concat'.

    Returns:
        JointMLP: The joint model
    """

    marginal_models = []

    for k in range(len(datas)):
        input_size = datas[k].shape[1]
        mmod = simple_FC(
            input_size = input_size, 
            hidden_sizes = hidden_sizes[k], 
            prediction_dim = prediction_dim,
            dropout = dropout,
            activation_fn = activation_fn
        )
        marginal_models.append(mmod)

    # joint model
    joint_model = JointMLP(marginal_models=marginal_models, hidden_dim=hidden_dim, activation_fn=activation_fn, combine_fn = combine_fn)

    return joint_model