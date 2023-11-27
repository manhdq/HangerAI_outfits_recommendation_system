# %%
import os.path as osp
import yaml
import pickle5 as pickle
import random
import importlib

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from Hash4AllFashion.utils.param import FashionDeployParam
from Hash4AllFashion.model import fashionnet
from app import prediction

importlib.reload(fashionnet)
importlib.reload(prediction)

# %%
project_dir = "/home/dungmaster/Projects/Machine Learning"
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "sample_images")

storage_path = osp.abspath(
    osp.join(
        project_dir,
        "HangerAI_outfits_recommendation_system",
        "storages",
        "hanger_apparels_100.pkl",
    )
)
storage_path

# # %% [markdown]
# # ### Load sample hash storage
# def load_hashes_storage(hfile):
#     pkl_file = open(hfile, "rb")
#     hashes_polyvore = pickle.load(pkl_file)[1]
#     return hashes_polyvore

# # %%
# hashes_polyvore = load_hashes_storage(storage_path)
# image_ids = list(hashes_polyvore.keys())
# print(len(image_ids))
# image_ids

# # %%
# hashes_polyvore

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

# %% [markdown]
# ### Load sample hash storage

# %%
hashes_polyvore = model.storage[1]
image_ids = list(hashes_polyvore.keys())
print(len(image_ids))
image_ids

# %%
sample = random.sample(hashes_polyvore.items(), 1)
sample

# %%
item_id = random.sample(image_ids, 1)[0]
img_path = osp.join(image_dir, item_id)
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(
    lci_v,
    lci_s,
    bci_v,
    bci_s,
    cate_prediction,
) = model.extract_feats_from_one_sample(image)
cate_prediction

# %%
plt.imshow(image)

# %% [markdown]
# ### Extract features and save to storage
apparel_feature_vecs = {}

for item_id in tqdm(image_ids):
    img_path = osp.join(image_dir, item_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        "new_hash_apparels_100.pkl",
    )
)
new_storage_path

# %%
with open(new_storage_path, "wb") as f:
    pickle.dump(new_hashes_polyvore, f)
    f.close()

# %%
