# %%
import sys
import os.path as osp
import yaml
import random
from PIL import Image
from pprint import pprint

from tqdm import tqdm
import importlib

sys.path += [".."]

from app.prediction import Pipeline
from infer_utils import load_json
from Hash4AllFashion.utils.param import FashionDeployParam


importlib.reload(file)

# TODO: create hanger_apparels_100.pkl file with semantic feature vector
# %% LOAD DATA
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "images")
disjoint_test_dataf = osp.join(data_dir, "disjoint", "test.json")
nondisjoint_test_dataf = osp.join(data_dir, "nondisjoint", "test.json")

# %%
disjoint_test_data = load_json(disjoint_test_dataf, verbose=True)
nondisjoint_test_data = load_json(nondisjoint_test_dataf, verbose=True)

# %%
all_items_ids_1 = []
for set_info in disjoint_test_data:
    set_items = set_info["items"]
    item_ids = [item["item_id"] for item in set_items]
    all_items_ids_1 += item_ids

all_items_ids_1 = set(all_items_ids_1)

all_items_ids_2 = []
for set_info in disjoint_test_data:
    set_items = set_info["items"]
    item_ids = [item["item_id"] for item in set_items]
    all_items_ids_2 += item_ids

all_items_ids_2 = set(all_items_ids_2)

all_items_ids = list(all_items_ids_1) + list(all_items_ids_2)
print(len(all_items_ids))
print(type(all_items_ids))

# %% LOAD MODEL
config_path = "../Hash4AllFashion/configs/deploy/FHN_VSE_T3_visual.yaml"
storage_path = "../storages/hanger_apparels_100.pkl"

with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

# %%
model = Pipeline(config, storage_path)
model.net

# %%
total_params = sum(p.numel() for p in model.net.parameters())
print(f"Number of parameters: {total_params}")

# %% EMBED IMAGES
rand_idx = random.randint(0, len(all_items_ids) - 1)
img_path = osp.join(image_dir, all_items_ids[rand_idx] + ".jpg")
image = Image.open(img_path)
(
    lci_v,
    lci_s,
    bci_v,
    bci_s,
    cate_prediction,
) = model.extract_feats_from_one_sample(image)
lci_s, lci_v, bci_s, bci_v, cate_prediction

# %%
apparel_feature_vecs = {}

for item_id in tqdm(all_items_ids[:10000]):
    img_name = item_id + ".jpg"
    img_path = osp.join(image_dir, img_name)
    image = Image.open(img_path)
    (
        lci_v,
        lci_s,
        bci_v,
        bci_s,
        cate_prediction,
    ) = model.extract_feats_from_one_sample(image)
    apparel_feature_vecs[img_name] = {
        "lci_v": lci_v,
        "lci_s": lci_s,
        "bci_v": bci_v,
        "bci_s": bci_s,
    }

# %%
feats = {1: apparel_feature_vecs}
file.to_pkl("../test.pkl", feats)

# %%
feat = model.storage[1]
first_pair = next(iter(feat.items()))
print(f"First Pair:")
key, val = first_pair
print("Key:", key)
print("Value:")
pprint(val)

# %%
apparel_feature_vecs = {}

for item_id in tqdm(all_items_ids[:10000]):
    img_name = item_id + ".jpg"
    img_path = osp.join(image_dir, img_name)
    image = Image.open(img_path)
    (
        lci_v,
        lci_s,
        bci_v,
        bci_s,
        cate_prediction,
    ) = model.extract_feats_from_one_sample(image)
    apparel_feature_vecs[img_name] = {
        "lci_v": lci_v,
        "lci_s": lci_s,
        "bci_v": bci_v,
        "bci_s": bci_s,
    }
