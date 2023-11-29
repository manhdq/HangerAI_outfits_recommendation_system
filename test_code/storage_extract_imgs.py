# %%
import os.path as osp
import yaml
import pickle5 as pickle
import random
import importlib

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import torch

from Hash4AllFashion.utils.param import FashionDeployParam
from Hash4AllFashion.model import fashionnet
from app import prediction

from reproducible_code.tools import io

importlib.reload(fashionnet)
importlib.reload(prediction)
importlib.reload(io)

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning"
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
visual_encoding = io.load_pickle(osp.join(data_dir, "visual_encoding.pkl"))

# %% [markdown]
# ### Load sample hash storage
storage_path = osp.abspath(
    osp.join(
        project_dir,
        "HangerAI_outfits_recommendation_system",
        "storages",
        "hanger_apparels_100.pkl",
    )
)
storage_path

# %% [markdown]
### Load model
config_path = "../Hash4AllFashion/configs/deploy/FHN_VSE_T3_visual_new.yaml"

with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

# %%
model = prediction.Pipeline(config, storage_path)
model.net

# %%
total_params = sum(p.numel() for p in model.net.parameters())
print(f"Number of parameters: {total_params}")

# %%
visual_encoding

# %% [markdown]
# ### Load sample hash storage

# %%
hashes_polyvore = model.storage[1]
image_ids = list(hashes_polyvore.keys())
print(len(image_ids))
image_ids

# # %%
# sample = random.sample(hashes_polyvore.items(), 1)
# sample

# # %%
# item_id = random.sample(image_ids, 1)[0]
# img_path = osp.join(image_dir, item_id)
# image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# (
#     lci_v,
#     lci_s,
#     bci_v,
#     bci_s,
#     cate_prediction,
# ) = model.extract_feats_from_one_sample(image)
# cate_prediction

# # %%
# plt.imshow(image)

# # %% [markdown]
# # ### Extract features and save to storage
# apparel_feature_vecs = {}

# for item_id in tqdm(image_ids):
#     img_path = osp.join(image_dir, item_id)
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     (
#         lci_v,
#         lci_s,
#         bci_v,
#         bci_s,
#         cate_prediction,
#     ) = model.extract_feats_from_one_sample(image)
#     apparel_feature_vecs[item_id] = {
#         "lci_v": lci_v,
#         "lci_s": lci_s,
#         "bci_v": bci_v,
#         "bci_s": bci_s,
#     }

# # %%
# print(len(apparel_feature_vecs))
# random.sample(apparel_feature_vecs.items(), 1)

# # %%
# new_hashes_polyvore = {1: apparel_feature_vecs}
# new_hashes_polyvore

# # %%
# new_storage_path = osp.abspath(
#     osp.join(
#         project_dir,
#         "HangerAI_outfits_recommendation_system",
#         "storages",
#         "hash_apparels_11_29.pkl",
#     )
# )
# new_storage_path

# # %%
# io.save_pickle(new_hashes_polyvore, new_storage_path)

# %% [markdown]
# ### Extract features from embeddings and save to storage using 
apparel_feature_vecs = {}

for item_id in tqdm(image_ids):
    img_encoding = torch.from_numpy(visual_encoding[item_id][0])
    (
        lci_v,
        lci_s,
        bci_v,
        bci_s,
        cate_prediction,
    ) = model.extract_feats_from_one_sample(img_encoding)
    apparel_feature_vecs[item_id] = {
        "lci_v": lci_v,
        "lci_s": lci_s,
        "bci_v": bci_v,
        "bci_s": bci_s,
    }
    # break

# %%
print(len(apparel_feature_vecs))
random.sample(apparel_feature_vecs.items(), 1)

# %%
new_hashes_polyvore = {1: apparel_feature_vecs}
new_hashes_polyvore

# %%
new_storage_path = osp.abspath(
    osp.join(
        project_dir,
        "HangerAI_outfits_recommendation_system",
        "storages",
        "hash_apparels_11_29.pkl",
    )
)
new_storage_path

# %%
io.save_pickle(new_hashes_polyvore, new_storage_path)

# %%
