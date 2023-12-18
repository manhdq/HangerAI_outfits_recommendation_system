# %%
import os
import os.path as osp
from glob import glob
import yaml
import random
import importlib

from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from reproducible_code.tools import io, plot

importlib.reload(io)

# %% [markdown]
# ### Load all embedded data
data_dir = "../../data"

item_infos = io.load_csv(osp.join(data_dir, "filter_result.csv"))
embeddings = np.load(osp.join(data_dir, "embedding.npy"))
print(len(embeddings))

# %%
print(item_infos["cate"].unique())
item_infos.head()

# %%
image_fpaths = glob(osp.join(data_dir, "chatbot_images", "*.jpg"))
len(image_fpaths)

# %%
img_ids = [osp.basename(p).split('.')[0] for p in image_fpaths]
len(img_ids)

# %%
embed_img_ids = item_infos.id.tolist()
embed_img_ids

# %%
missing = {i: img_id for i, img_id in enumerate(embed_img_ids) if img_id not in img_ids}
missing

# %%
miss_id = list(missing.values())[0]
miss_idx = list(missing.keys())[0]
miss_idx

# %%
item_infos.iloc[miss_idx, 2] = ''
item_infos.iloc[miss_idx]

# %%
io.save_csv(item_infos, osp.join(data_dir, "filter_result.csv"))

# %% [markdown]
### Test filter by cate
embeddings.shape

# %%
search_cate = "top"
idxs = item_infos[item_infos["cate"] == search_cate].index.tolist()
len(idxs)

# %%
item_infos.iloc[idxs, 1].tolist()

# %%
embeddings[idxs, :].shape

# %%
