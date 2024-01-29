import os.path as osp
import requests
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

from io import BytesIO
import streamlit as st
from src.utils.utils import image_to_base64
from omegaconf import OmegaConf

import torch
from icecream import ic


def main(config_path: str):
    configs = OmegaConf.load(config_path)    
    
    ### Edit these params ###
    image_dir = osp.join(configs.path.data_dir_path, configs.path.image_dir)
    cates = configs.recommend.cates
    num_outfits = configs.recommend.num_outfits

    num_cols = len(cates)
    table_style = False


    ## Running the web app
    title = "Fashion Outfits Search App"
    st.set_page_config(page_title=title, layout="wide")
    st.header(title)

    ## Input search query
    prompt = st.text_input("Search: ")

    # Sidebar section for uploading files and providing a YouTube URL
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=None)
        st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

    names = []
    images = {}
    input_64s = []

    if uploaded_files:
        st.write(f"Number of files uploaded: {len(uploaded_files)}")
        for uploaded_file in uploaded_files:
            name = uploaded_file.name
            names.append(name)

            bytes_data = uploaded_file.read()    
            image = Image.open(uploaded_file)
            images[name] = image
            inp_64 = image_to_base64(image)
            input_64s.append(inp_64)

    if len(prompt) == 0:
        prompt = "wedding suit for men"


    ## Send request to outfit recommend api
    response = requests.post(
        url="http://127.0.0.1:3000/outfits_recommend_from_inputs/",
        json={
            "text": prompt,
            "name": names,
            "input64": input_64s,
        },
    )
    assert response.status_code == 200, response.status_code
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

                    # image_path = osp.join(image_dir, str(garm_id)+".jpg")
                    # image = Image.open(image_path)

                    image = images[garm_id]
                    image = expand2square_pil(image)

                    st.image(
                        image,
                        width=300
                    )
                    ind_garment_retrieved += 1

            ind_outfit += 1

            if ind_outfit >= num_outfits:
                break


if __name__ == "__main__":
    config_path = "configs/polyvore_outfit_recommend.yaml"
    main(config_path)
