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

from reproducible_code.tools import io

importlib.reload(io)

# %% [markdown]
### Load all labelled data
data_dir = "../../Data_processed"
full_csv = osp.join(data_dir, "labelled_full.csv")
df_full = io.load_csv(full_csv)
df_full.head()

# %%
saved_imgs_dir = "../../data/chatbot_images"
io.create_dir(saved_imgs_dir)

# %% [markdow]
### Save all images to dir
for idx, row in tqdm(df_full.iterrows()):
    item_id = row.id
    image = None
    for look in ["look", "front", "left", "right", "back"]:
        url = row[look]
        if type(url) == str:
            response = requests.get(url)
            if response.status_code == 200:
                content = BytesIO(response.content)
                try:
                    image = np.frombuffer(content.read(), np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    continue
            img_path = osp.join(saved_imgs_dir, item_id + f".jpg")
            cv2.imwrite(img_path, image)
            break

# %%
