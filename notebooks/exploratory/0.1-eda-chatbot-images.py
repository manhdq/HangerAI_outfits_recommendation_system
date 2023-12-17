# %%
import os
import os.path as osp
import requests
from io import BytesIO

import yaml
import random
import importlib

from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from reproducible_code.tools import io, plot

importlib.reload(io)

# %% [markdown]
# ### Load all labelled data
data_dir = "../../Data_processed"
dfs = [pd.read_csv(osp.join(data_dir, fn)) for fn in os.listdir(data_dir)]
df_full = pd.concat(dfs)
len(df_full)

# %%
df_full.columns

# %% [markdown]
# Remove "other" category
df_full = df_full[df_full["label"] != "other"].reset_index(drop=True)
len(df_full)

# %% [markdown]
# ### Display category distribution
df_full["label"].unique().tolist()

# %%
df_full = df_full[df_full["label"].notna()].reset_index(drop=True)
len(df_full)

# %%
len(df_full["label"].unique().tolist())

# %%
plot.plot_attribute_frequency(df_full, "label", 10, 5)

# %% [markdown]
# Retrieve some urls and plot some imgs
# %%
chosen_cates = ["top", "bottom", "footwear", "outerwear", "bag"]
df_full = df_full[df_full["label"].isin(chosen_cates)].reset_index(drop=True)
len(df_full)

# %%
num_sample = 25
count = 0

list_images = []
list_ids = []

while count < num_sample:
    imgl = []
    list_url = []
    item_look = []
    
    idx = random.randint(0, len(df_full)-1)
    row = df_full.iloc[idx]
    item_id = row["id"]

    for look in ["look", "front", "back", "left", "right"]:
        if type(row[look]) == str:
            list_url.append(row[look])
            item_look.append(item_id + ' ' + look)

    # if type(row["look"]) == str:
    #     list_url.append(row["look"])
    # if type(row["front"]) == str:
    #     list_url.append(row["front"])
    # if type(row["back"]) == str:
    #     list_url.append(row["back"])
    # if type(row["left"]) == str:
    #     list_url.append(row["left"])
    # if type(row["right"]) == str:
    #     list_url.append(row["right"])

    for url in list_url:
        response = requests.get(url)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
        imgl.append(image)

    count += len(imgl)
    print(count)
    list_ids.extend(item_look)
    list_images.extend(imgl)

# %%
len(list_images), len(list_ids)

# %%
plot.display_multiple_images(
    list_images,
    5,
    titles=list_ids,
    axes_pad=.8
)

# %% [markdow]
# ## Save for later use
full_csv = osp.join(data_dir, "labelled_full.csv")
io.save_csv(df_full, full_csv)

# %%
