import os.path as osp
import requests
from pprint import pprint
import pandas as pd
from PIL import Image
import streamlit as st

import torch
from reproducible_code.tools import image_io


### Edit these params ###
image_dir = "data/chatbot_images"
cates = ["top", "bottom", "bag", "outerwear", "footwear"]
num_cols = len(cates)
num_outfits = 4
table_style = False


## Running the web app
title = "Fashion Outfits Search App"
st.set_page_config(page_title=title, layout="wide")
st.header(title)


## Input search query
prompt = st.text_input("Search: ")
if len(prompt) == 0:
    prompt = "wedding suit for men"


## Send request to outfit recommend api
response = requests.post(
    url="http://127.0.0.1:3000/outfits_recommend_from_prompt/",
    json={"text": prompt},
)
json_response = response.json()
outfit_recommends = json_response["outfit_recommend"]


## Showcase api's reponse on web app
cols = st.columns(num_cols, )

## TODO
if table_style:
    outfit_df = pd.DataFrame(outfit_recommends)
    image_columns = {
        c: st.column_config.ImageColumn(
            help="Garment screenshot"
        )
        for c in cates
    }

    st.data_editor(
        outfit_df,
        width=1920,
        column_config=image_columns,
    )
else:
    ind_garment_retrieved = 0
    ind_outfit = 0

    for ind, outfit in enumerate(outfit_recommends):
        for cate in cates:
            garm_id = outfit[cate]

            ind_col = int(ind_garment_retrieved % num_cols)
            col = cols[ind_col]

            with col:
                if ind_outfit == 0:
                    st.header(cate)

                image = Image.open(osp.join(image_dir, garm_id+".jpg"))
                image = image_io.expand2square(image)
                
                st.image(
                    image,
                    # width=250
                )
                ind_garment_retrieved += 1

        ind_outfit += 1

        if ind_outfit >= num_outfits:
            break
