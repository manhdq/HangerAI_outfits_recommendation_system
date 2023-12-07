import sys
import os
import os.path as osp
import yaml
import logging
import pickle5 as pickle
import json
from collections import defaultdict
import numpy as np
from pprint import pprint
from icecream import ic

from fastapi import status, HTTPException, Depends, APIRouter, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import models, utils, prediction
from ..schemas import apparel as schemas
from ..database import get_db
from Hash4AllFashion_deploy.utils.param import FashionDeployParam
from Hash4AllFashion_deploy.utils.logger import Logger, config_log

sys.path += ["CapstoneProject"]

from apis.search_fclip import FashionRetrieval
from tools import load_json

router = APIRouter(prefix="/items", tags=["Apparels"])


# Hyperparams for outfit recommend
outfit_recommend_option = defaultdict(list)
# cates = ["top", "bottom", "bag", "outerwear", "full-body", "footwear", "accessory"]
cates = ["top", "bottom", "bag", "outerwear", "shoe"]

n_outfits = 4
chosen_cate = "dynamic"
mode = "each"  # get top k items from each category then compose an outfit
# mode = "all"  # get top k items from all categories then compose an outfit
empty_cate_extras = 5

if mode == "all":
    top_k = 20
elif mode == "each":
    top_k = 7


# Important functions
def get_logger(env, config):
    if env == "local":
        ##TODO: Modify this logger name
        logfile = config_log(
            stream_level=config.log_level, log_file=config.log_file
        )
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


def load_meta(metadata_file):
    metadata = load_json(metadata_file)
    return metadata


def get_category(path):
    image_id = name(path)
    category = metadata[image_id]["semantic_category"]
    if category != "outerwear":
        category = category[:-1]
    return image_id, category


name = lambda x: osp.basename(x).split(".")[0]

## Get config, pipeline
# config_path = "Hash4AllFashion_deploy/configs/deploy/FHN_VSE_T3_visual_new.yaml"
config_path = "Hash4AllFashion_deploy/configs/deploy/FHN_VOE_T3_fashion32.yaml"
env = "colab"

##TODO: Delete logger
with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

pipeline = prediction.Pipeline(
    config,
    storage_path=config.hash_storage
)

## Important paths to data
root_dir = "/home/dungmaster/Projects/Machine Learning/HangerAI_outfits_recommendation_system/CapstoneProject"
embeddings_dir = osp.join(root_dir, "model_embeddings", "polyvore")

data_dir = "/home/dungmaster/Datasets/polyvore_outfits"
image_dir = osp.join(data_dir, "sample_images")
metadata_file = osp.join(data_dir, "polyvore_item_metadata.json")
item2cate_fpath = osp.join(data_dir, "item_cate_502.json")

embeddings_file = None
if mode == "all":
    embeddings_file = osp.join(root_dir, "model_embeddings", "polyvore_502.txt")

## Get data, embedding for fashion retrieval

image_embeddings = None
save_embeddings = False

image_paths = [osp.join(image_dir, str(name)) for name in list(pipeline.storage[1].keys())]
metadata = load_meta(metadata_file)
item_cate_map = json.load(open(item2cate_fpath, 'r'))

if embeddings_file is not None and osp.exists(embeddings_file):
    image_embeddings = np.loadtxt(embeddings_file)

ret = FashionRetrieval(
    image_embeddings=image_embeddings
)


#### GET ####
@router.get("/{user_id}/{item_name}", response_model=schemas.Apparel)
async def get_items(
    user_id: int, item_name: str, db: Session = Depends(get_db)
):
    item = (
        db.query(models.Apparel)
        .filter(
            models.Apparel.user_id == user_id,
            models.Apparel.name == f"{item_name}.jpg",
        )
        .first()
    )
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"item with name {item_name} was not found",
        )

    return item


#### POST ####
@router.post(
    "/{user_id}/upload_item",
    response_model=schemas.Apparel,
    status_code=status.HTTP_201_CREATED,
)
async def create_item(
    user_id: int,
    item_input: schemas.ApparelCreateInput,
    db: Session = Depends(get_db),
):
    item_name = item_input.name
    input64 = item_input.input64
    price = item_input.price

    # Check if the image exist in database according to user_id
    item = (
        db.query(models.Apparel)
        .filter(
            models.Apparel.user_id == user_id,
            models.Apparel.name == f"{item_name}.jpg",
        )
        .first()
    )
    if item is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"apparel {item_name} already exists in database",
        )

    img_pil = utils.base64_to_image(input64)
    (
        lci_v,
        lci_s,
        bci_v,
        bci_s,
        cate_prediction,
    ) = pipeline.extract_feats_from_one_sample(img_pil)
    w, h = img_pil.size

    storage = load_storage(pseudo_s3_storage_file)

    storage[user_id] = storage.get(user_id, dict())
    # assert item_name not in storage[user_id]
    storage[user_id][f"{item_name}.jpg"] = {
        "lci_v": lci_v,
        "lci_s": lci_s,
        "bci_v": bci_v,
        "bci_s": bci_s,
        "base64": input64,
    }
    save_storage(pseudo_s3_storage_file, storage)
    del storage
    pipeline._load_storage()

    new_item = models.Apparel(
        name=f"{item_name}.jpg",
        width=w,
        height=h,
        category=cate_prediction,
        price=price,
        user_id=user_id,
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)

    return new_item


#### PUT ####
@router.put("/{user_id}/{item_name}", response_model=schemas.Apparel)
def update_post(
    user_id: int,
    item_name: str,
    item: schemas.ApparelUpdate,
    db: Session = Depends(get_db),
):
    item_query = db.query(models.Apparel).filter(
        models.Apparel.user_id == user_id,
        models.Apparel.name == f"{item_name}.jpg",
    )

    if item_query.first() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"item with name {item_name} was not found",
        )

    # Check if change item name not in database
    if item_name != item.name.split(".")[0]:
        check_item_query = db.query(models.Apparel).filter(
            models.Apparel.user_id == user_id, models.Apparel.name == item.name
        )

        if check_item_query.first() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"item with name {item_name.split('.')[0]} was already exists",
            )

    storage = load_storage(pseudo_s3_storage_file)
    assert user_id in storage
    assert f"{item_name}.jpg" in storage[user_id]
    if item_name != item.name.split(".")[0]:
        # Change storage
        storage[user_id][item.name] = storage[user_id][
            f"{item_name}.jpg"
        ].copy()
        del storage[user_id][f"{item_name}.jpg"]

    save_storage(pseudo_s3_storage_file, storage)
    del storage
    pipeline._load_storage()

    item_query.update(item.dict(), synchronize_session=False)
    db.commit()

    # Get new item according to change
    item_query = db.query(models.Apparel).filter(
        models.Apparel.user_id == user_id, models.Apparel.name == item.name
    )

    return item_query.first()


#### DELETE ####
@router.delete(
    "/{user_id}/{item_name}", status_code=status.HTTP_204_NO_CONTENT
)
def delete_post(user_id: int, item_name: str, db: Session = Depends(get_db)):
    item_query = db.query(models.Apparel).filter(
        models.Apparel.user_id == user_id,
        models.Apparel.name == f"{item_name}.jpg",
    )

    if item_query.first() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"item with name {item_name} was not found",
        )

    storage = load_storage(pseudo_s3_storage_file)
    assert user_id in storage
    assert f"{item_name}.jpg" in storage[user_id]
    del storage[user_id][f"{item_name}.jpg"]

    save_storage(pseudo_s3_storage_file, storage)
    del storage
    pipeline._load_storage()

    item_query.delete(synchronize_session=False)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{user_id}/outfits_recommend/")
def outfits_recommend(
    user_id: int,
    outfit_recommend_option: schemas.OutfitsRecommendation,
    db: Session = Depends(get_db),
):
    chosen = {}
    recommend_choices = []

    for cate in ["top", "bottom", "bag", "outerwear", "shoe"]:
        if isinstance(outfit_recommend_option.dict()[cate], str):
            img_name = outfit_recommend_option.dict()[cate]
            if img_name[-4:] != ".jpg":
                img_name = img_name + ".jpg"
            item = (
                db.query(models.Apparel)
                .filter(
                    models.Apparel.user_id == user_id,
                    models.Apparel.name == img_name,
                )
                .first()
            )
            if item is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"item with name {img_name.split('.')[0]} was not found",
                )
            chosen[cate] = img_name

        else:  # int
            if outfit_recommend_option.dict()[cate] == 1:  # Enable
                recommend_choices.append(cate)

    outputs = pipeline.outfit_recommend(
        chosen=chosen,
        recommend_choices=recommend_choices,
        db=db,
        user_id=user_id,
    )
    return outputs


@router.post("/{user_id}/outfits_recommend_from_chosen/")
def outfits_recommend_from_chosen(
    user_id: int,
    outfit_recommend_option: schemas.OutfitsRecommendation,
    db: Session = Depends(get_db),
):
    chosen = defaultdict(list)
    chosen_cate = "top"
    outfit_recommend_option = outfit_recommend_option.dict()

    for cate in ["top", "bottom", "bag", "outerwear", "shoe"]:
        cate_items = outfit_recommend_option[cate]
        if isinstance(cate_items, list):
            for img_name in cate_items:
                if img_name[-4:] != ".jpg":
                    img_name = img_name + ".jpg"
                item = (
                    db.query(models.Apparel)
                    .filter(
                        models.Apparel.user_id == user_id,
                        models.Apparel.name == img_name,
                    )
                    .first()
                )
                if item is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"item with name {img_name.split('.')[0]} was not found",
                    )
                chosen[cate].append(img_name)

        # if len(cate_items) == 0:
        #     chosen[cate] = "all"

    list_given_items = chosen[chosen_cate]
    dict_given_items = [{chosen_cate: item} for item in list_given_items]
    recommend_choices = {k: v for k, v in chosen.items() if k != chosen_cate}

    outputs = pipeline.outfit_recommend_from_chosen(
        given_items=dict_given_items,
        recommend_choices=recommend_choices,
        db=db,
        user_id=user_id,
    )
    return outputs





@router.post("/{user_id}/outfits_recommend_from_prompt/")
def outfits_recommend_from_prompt(
    user_id: int,
    prompt: schemas.TextInput,
    db: Session = Depends(get_db),
):
    processed_text = prompt.text.lower()

    # Retrieve items matching the prompt
    if mode == "all":
        found_image_paths, text_embedding = ret.retrieve(
            query=processed_text,
            image_paths=image_paths,
            save_embeddings=save_embeddings,
            embeddings_file=embeddings_file
        )
        prompt_matched_image_paths = found_image_paths[:top_k]
        for image_path in prompt_matched_image_paths:
            image_id, category = get_category(image_path)
            outfit_recommend_option[category].append(image_id)

        # Add items in categories in which there is no item
        # pprint(outfit_recommend_option)
        for cate in cates:
            count = 0
            if cate not in list(outfit_recommend_option.keys()):
                for image_path in found_image_paths[top_k:]:
                    image_id, category = get_category(image_path)
                    if category == cate:
                        outfit_recommend_option[cate].append(image_id)
                        count += 1
                    if count >= empty_cate_extras:
                        break            

    elif mode == "each":
        for search_cate in cates:
            embeddings_file = osp.join(embeddings_dir, f"polyvore_{search_cate}_502.txt")
            image_paths = [
                osp.join(image_dir, img_fn) for img_fn in item_cate_map[search_cate]
            ]
            found_image_paths, text_embedding = ret.retrieve(
                query=processed_text,
                image_paths=image_paths,
                # save_embeddings=save_embeddings,
                embeddings_file=embeddings_file
            )
            prompt_matched_image_paths = found_image_paths[:top_k]
            for image_path in prompt_matched_image_paths:
                image_id = name(image_path)
                outfit_recommend_option[search_cate].append(image_id)    

    # Compose outfits from retrieved items
    chosen = defaultdict(list)

    if not pipeline.use_outfit_semantic:
        text_embedding = None

    for cate in cates:
        cate_items = outfit_recommend_option[cate]
        if isinstance(cate_items, list):
            for img_name in cate_items:
                if img_name[-4:] != ".jpg":
                    img_name = img_name + ".jpg"
                item = (
                    db.query(models.Apparel)
                    .filter(
                        models.Apparel.user_id == user_id,
                        models.Apparel.name == img_name,
                    )
                    .first()
                )
                if item is None:
                    # raise HTTPException(
                    #     status_code=status.HTTP_404_NOT_FOUND,
                    #     detail=f"item with name {img_name.split('.')[0]} was not found",
                    # )
                    print(
                        f"item with name {img_name.split('.')[0]} was not found!"
                    )
                    continue
                else:
                    chosen[cate].append(img_name)

    # Recommend outfit not from just top items
    # of a particular category exclusively
    # but from top results retrieved of previous model
    outputs = {
        "first_item_cate": [],
        "outfit_recommend": []
    }
    if chosen_cate == "dynamic":
        for ind, image_path in enumerate(found_image_paths[:n_outfits]):
            image_id, category = get_category(image_path)
            # list_given_items = chosen[category]
            # dict_given_items = [{category: item} for item in list_given_items]
            dict_given_items = [{category: image_id + ".jpg"}]
            recommend_choices = {
                k: v for k, v in chosen.items() if k != category
            }
            output = pipeline.outfit_recommend_from_chosen(
                given_items=dict_given_items,
                recommend_choices=recommend_choices,
                db=db,
                user_id=user_id,
                outfit_semantic=text_embedding
            )["outfit_recommend"][0]

            # Preprocess the output
            output = {cate: osp.join(image_dir, name) for cate, name in output.items()}
            # output["Outfit Number"] = ind
            
            # Remove duplicate outfits
            if output not in outputs["outfit_recommend"]:
                outputs["first_item_cate"].append(category)
                outputs["outfit_recommend"].append(output)
    else:
        list_given_items = chosen[chosen_cate]
        dict_given_items = [{chosen_cate: item} for item in list_given_items]
        recommend_choices = {
            k: v for k, v in chosen.items() if k != chosen_cate
        }

        outputs = pipeline.outfit_recommend_from_chosen(
            given_items=dict_given_items,
            recommend_choices=recommend_choices,
            db=db,
            user_id=user_id,
        )

    outfit_recommend_option.clear()

    return outputs
