import torch.nn as nn

# need to wrap the model in this class to get around some issues with the SHAP package
class JointMLPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *datas):
        yhat, _, _, _ = self.model(datas)
        return yhat
