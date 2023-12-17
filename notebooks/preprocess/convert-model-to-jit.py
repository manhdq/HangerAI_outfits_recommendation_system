# %%
import os.path as osp
import yaml
import random
import importlib

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from Hash4AllFashion_deploy.utils.param import FashionDeployParam
from Hash4AllFashion_deploy.model import fashionnet
from app import prediction

from reproducible_code.tools import io

importlib.reload(fashionnet)
importlib.reload(prediction)
importlib.reload(io)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %% [markdown]
# ### Load sample hash storage
storage_path = "../../storages/hanger_apparels_100.pkl"
storage_path

# %% [markdown]
# ## Load model
config_path = "../../Hash4AllFashion_deploy/configs/deploy/FHN_VOE_T3_fashion32.yaml"

with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

# %%
model = prediction.Pipeline(config, storage_path)
model.net

# %% [markdown]
# ### Export to TorchScript
# Switch the model to eval mode
model = model.net
model.eval()

# %%
input_dict = {}
outf_s_out = torch.rand(1, 512).to(device)
imgs = torch.rand(3, 3, 224, 224).to(device)
inputs = (imgs, outf_s_out)

# %%
model(*inputs)

# %%
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (imgs, outf_s_out), check_trace=False)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.pt")

# %% [markdown]
# ### Save only CoreMat module
core = model.core
core_pth =  "../../Hash4AllFashion_deploy/checkpoints/11_26_23/core_best_11_26.pt"
torch.save(core.state_dict(), core_pth)

# %% [markdown]
# ### Load again and check if weight is equal to original model's weight
class CoreMat(nn.Module):
    """Weighted hamming similarity."""

    def __init__(self, dim, weight=1.0):
        """Weights for this layer that is drawn from N(mu, std)."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights(weight)

    def init_weights(self, weight):
        """Initialize weights."""
        self.weight.data.fill_(weight)

    def forward(self, x):
        """Forward."""
        return torch.mul(x, self.weight)

    def __repr__(self):
        """Format string for module CoreMat."""
        return self.__class__.__name__ + "(dim=" + str(self.dim) + ")"

core = nn.ModuleList(
    [
        CoreMat(128, 1.0),
        CoreMat(128, 1.5),
    ]
)
state_dict = torch.load(core_pth)
core.load_state_dict(state_dict)

# %%
for name, param in state_dict.items():
    if name in model.core.state_dict().keys():
        param = param.data
        print((state_dict[name] == param).all())

# %%
