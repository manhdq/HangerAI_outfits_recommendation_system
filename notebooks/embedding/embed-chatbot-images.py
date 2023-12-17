# %%
import os
import os.path as osp
import yaml
import pickle5 as pickle
import random
import importlib

import requests
from io import BytesIO

from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch

from Hash4AllFashion_deploy.utils.param import FashionDeployParam
from Hash4AllFashion_deploy.model import fashionnet
from app import prediction

from reproducible_code.tools import io

importlib.reload(fashionnet)
importlib.reload(prediction)
importlib.reload(io)

# %% [markdown]
### Load all labelled data
data_dir = "../../Data_processed"
full_csv = osp.join(data_dir, "labelled_full.csv")
df_full = io.load_csv(full_csv)
df_full.head()

# %% [markdown]
## Fashion Hashnet

# %% [markdown]
### Load sample hash storage
storage_path = "../../storages/random.pkl"  # dummy storage
storage_path

# %% [markdown]
### Load model
config_path = "../../Hash4AllFashion_deploy/configs/deploy/FHN_VOE_T3_fashion32.yaml"

with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

# %%
model = prediction.Pipeline(config, storage_path)
model.net

# %% [markdown]
### Extract features and save to storage
apparel_feature_vecs = {}

# %%
images = []
for idx, row in tqdm(df_full.iterrows()):
    item_id = row.id
    if item_id in apparel_feature_vecs:
        continue
    image = None
    for look in ["look", "front", "left", "right", "back"]:
        url = row[look]
        if type(url) == str:
            response = requests.get(url)
            if response.status_code == 200:
                content = BytesIO(response.content)
                # image = Image.open(content)
                try:
                    image = np.frombuffer(content.read(), np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    continue
            break

    # images.append(image)

    if image is not None:
        (
            lci_v,
            lci_s,
            bci_v,
            bci_s,
            cate_prediction,
        ) = model.extract_feats_from_one_sample(image)
        apparel_feature_vecs[item_id] = {
            "lci_v": lci_v,
            "lci_s": lci_s,
            "bci_v": bci_v,
            "bci_s": bci_s,
        }

# # %%
# plt.imshow(random.sample(images, 1)[0])

# %%
print(len(apparel_feature_vecs))
random.sample(apparel_feature_vecs.items(), 1)

# %%
new_hashes_polyvore = {1: apparel_feature_vecs}
new_hashes_polyvore

# %%
new_storage_path = "../../storages/hash_apparels_chatbot.pkl"
new_storage_path

# %%
# io.save_pickle(new_hashes_polyvore, new_storage_path)
new_hashes_polyvore = io.load_pickle(new_storage_path)
len(new_hashes_polyvore[1])

# %% [markdown]
## Fashion CLIP
