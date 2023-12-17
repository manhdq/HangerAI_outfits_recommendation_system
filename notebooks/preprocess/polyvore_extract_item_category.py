# %%
import os
import os.path as osp
import yaml
import pickle5 as pickle
import random
import importlib
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from reproducible_code.tools import io

# %% [markdown]
# ### Load image and metadata
data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
images = os.listdir(
    osp.join(data_dir, "sample_images")
)
print(len(images))
item_metadata = io.load_json(
    osp.join(data_dir, "polyvore_item_metadata.json")
)
categories = io.load_csv(
    osp.join(data_dir, "categories.csv")
)
categories.head()

# %% [markdown]
# ### Create item-category dataframe
df_item_cate = pd.DataFrame(columns=["path", "category"])

# %%
for image in tqdm(images):
    img_id = image.split('.')[0]
    cate_id = int(item_metadata[img_id]["category_id"])
    cate = categories[categories['2'] == cate_id]["Unnamed: 2"].iloc[0]
    df_item_cate.loc[len(df_item_cate.index)] = [image, cate]

print(len(df_item_cate))
df_item_cate.head()

# %%
df_item_cate["category"].unique()

# %%
sample_csv = osp.join(data_dir, "item_cate_502.csv")
io.to_csv(sample_csv, df_item_cate)

# %%
item_cates = defaultdict(list)

for image in tqdm(images):
    img_id = image.split('.')[0]
    cate_id = int(item_metadata[img_id]["category_id"])
    cate = categories[categories['2'] == cate_id]["Unnamed: 2"].iloc[0]
    if cate != "outerwear":
        cate = cate[:-1]
    item_cates[cate].append(image)
    # item_cates[image] = cate

item_cates

# %%
sample_json = osp.join(data_dir, "item_cate_502.json")
io.save_json(item_cates, sample_json)

# %%
for k, v in item_cates.items():
    print(k, len(v))

# %%
