import base64
import time
import logging
import yaml
import pickle
import os
import warnings
import psycopg2
from typing import Union, List, ByteString
from typing_extensions import Annotated
from psycopg2.extras import RealDictCursor
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Response, status, HTTPException, \
                    UploadFile, Body, File, Query
from pydantic import BaseModel

from Hash4AllFashion.utils.param import FashionDeployParam
from Hash4AllFashion.utils.logger import Logger, config_log
from utils import base64_to_image
from prediction import Pipeline


def get_logger(env, config):
    if env == "local":
        ##TODO: Modify this logger name
        logfile = config_log(stream_level=config.log_level, log_file=config.log_file)
        logger = logging.getLogger("polyvore")
        logger.info("Logging to file %s", logfile)
    elif env == "colab":
        logger = Logger(config)  # Normal logger
        logger.info(f"Logging to file {logger.logfile}")

    return logger


def load_storage(pkl_file):
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as file:
            loaded_data = pickle.load(file)
        return loaded_data
    return dict()


def save_storage(pkl_file, storage):
    with open(pkl_file, "wb") as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)


while True:
    try:
        conn = psycopg2.connect(host='localhost', 
                                database='fastapi', 
                                user='demo', 
                                password="210301",
                                cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        print("Database connection was successful")
        break
    except Exception as error:
        print("Connection to database failed")
        print("Error:", error)
        time.sleep(2)


## Get config, pipeline
config_path = "./Hash4AllFashion/configs/deploy/FHN_VSE_T3_visual.yaml"
env = "colab"
table_name = "hanger_apparels_100"
pseudo_s3_storage_file = f"storages/{table_name}.pkl"

##TODO: Delete logger
with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

pipeline = Pipeline(config,
                    cursor,
                    table_name=table_name,
                    storage_path=pseudo_s3_storage_file)


class Apparel(BaseModel):
    name: str
    width: int
    height: int
    category: str


class OutfitRecommendOption(BaseModel):
    img_name: str
    recommend_top: bool = True
    recommend_bottom: bool = True
    recommend_bag: bool = True
    recommend_outerwear: bool = True
    recommend_shoe: bool = True


app = FastAPI()


#### GET ####
@app.get("/")
async def root():
    return {"message": "Welcome to HangerAI Outfit Recommendation"}

@app.get("/items")
async def get_items():
    cursor.execute("""SELECT * FROM hanger_apparels_100""")
    items = cursor.fetchall()
    return {"data": items}

# @app.get("/items/{img_name}")
# async def get_item(img_name):
#     cursor.execute("""SELECT * FROM WHERE""")


#### POST ####
@app.post("/upload_item", status_code=status.HTTP_201_CREATED)
async def create_item(file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    img_pil = base64_to_image(base64_image)
    lci_v, lci_s, bci_v, bci_s, cate_prediction = \
            pipeline.extract_feats_from_one_sample(img_pil)
    w, h = img_pil.size
    
    apparel = Apparel(name=file.filename,
                    width=w, height=h,
                    category=cate_prediction)

    storage = load_storage(pseudo_s3_storage_file)
    ## Check if image exists in storage
    if file.filename in storage:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"apparel {file.filename} already exists in database")

    storage[file.filename] = {
        "lci_v": lci_v,
        "lci_s": lci_s,
        "bci_v": bci_v,
        "bci_s": bci_s,
        "base64": base64_image,
    }
    save_storage(pseudo_s3_storage_file, storage)
    del storage
    pipeline._load_storage()

    cursor.execute("""INSERT INTO hanger_apparels_100 (name, width, height, category)
                    VALUES (%s, %s, %s, %s) RETURNING *""",
                    (apparel.name, apparel.width, apparel.height, apparel.category))
    apparel = cursor.fetchone()

    conn.commit()

    return {"data": apparel}

@app.post("/outfits_recommend/items/")
def outfits_recommend(outfit_recommend_option: OutfitRecommendOption):
    img_name = outfit_recommend_option.img_name

    if img_name[-4:] != ".jpg":
        img_name = img_name + ".jpg"
        
    cursor.execute("""SELECT * FROM hanger_apparels_100 WHERE name = %s""", [img_name])
    apparel = cursor.fetchone()
    if apparel is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"apparel with name {img_name} was not found")

    chosen = {apparel['category']: img_name}
    recommend_choices = []
    if outfit_recommend_option.recommend_top: recommend_choices.append("top");
    if outfit_recommend_option.recommend_bottom: recommend_choices.append("bottom");
    if outfit_recommend_option.recommend_bag: recommend_choices.append("bag");
    if outfit_recommend_option.recommend_outerwear: recommend_choices.append("outerwear");
    if outfit_recommend_option.recommend_shoe: recommend_choices.append("shoe");
    recommend_choices = [choice for choice in recommend_choices if choice not in chosen]
    
    outputs = pipeline.outfit_recommend(chosen=chosen, recommend_choices=recommend_choices)
    return {"data": outputs}