import os
import yaml
import logging
import pickle5 as pickle

from fastapi import status, HTTPException, Depends, APIRouter, Response
from sqlalchemy.orm import Session

from .. import models, utils, prediction
from ..schemas import apparel as schemas
from ..database import get_db
from Hash4AllFashion.utils.param import FashionDeployParam
from Hash4AllFashion.utils.logger import Logger, config_log

router = APIRouter(prefix="/items", tags=["Apparels"])


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


## Get config, pipeline
config_path = "Hash4AllFashion/configs/deploy/FHN_VSE_T3_visual.yaml"
env = "colab"
table_name = "hanger_apparels_100"
pseudo_s3_storage_file = f"storages/{table_name}.pkl"

##TODO: Delete logger
with open(config_path, "r") as f:
    kwargs = yaml.load(f, Loader=yaml.FullLoader)
config = FashionDeployParam(**kwargs)

pipeline = prediction.Pipeline(config,
                            storage_path=pseudo_s3_storage_file)


#### GET ####
@router.get("/{user_id}/{item_name}", response_model=schemas.Apparel)
async def get_items(user_id: int, item_name: str, db: Session = Depends(get_db)):
    item = db.query(models.Apparel).filter(models.Apparel.user_id==user_id, 
                                models.Apparel.name==f"{item_name}.jpg").first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"item with name {item_name} was not found")

    return item


#### POST ####
@router.post("/{user_id}/upload_item", response_model=schemas.Apparel, status_code=status.HTTP_201_CREATED)
async def create_item(user_id: int, item_input: schemas.ApparelCreateInput, db: Session = Depends(get_db)):
    item_name = item_input.name
    input64 = item_input.input64
    price = item_input.price

    # Check if the image exist in database according to user_id
    item = db.query(models.Apparel).filter(models.Apparel.user_id==user_id, 
                                models.Apparel.name==f"{item_name}.jpg").first()
    if item is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"apparel {item_name} already exists in database")

    img_pil = utils.base64_to_image(input64)
    lci_v, lci_s, bci_v, bci_s, cate_prediction = \
            pipeline.extract_feats_from_one_sample(img_pil)
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

    new_item = models.Apparel(name=f"{item_name}.jpg",
                            width=w,
                            height=h,
                            category=cate_prediction,
                            price=price,
                            user_id=user_id)
    db.add(new_item)
    db.commit()
    db.refresh(new_item)

    return new_item


#### PUT ####
@router.put("/{user_id}/{item_name}", response_model=schemas.Apparel)
def update_post(user_id: int, item_name: str, item: schemas.ApparelUpdate, db: Session = Depends(get_db)):
    item_query = db.query(models.Apparel).filter(models.Apparel.user_id==user_id, models.Apparel.name==f"{item_name}.jpg")

    if item_query.first() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"item with name {item_name} was not found")

    # Check if change item name not in database
    if item_name != item.name.split('.')[0]:
        check_item_query = db.query(models.Apparel).filter(models.Apparel.user_id==user_id, models.Apparel.name==item.name)

        if check_item_query.first() is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"item with name {item_name.split('.')[0]} was already exists")

    storage = load_storage(pseudo_s3_storage_file)
    assert user_id in storage
    assert f"{item_name}.jpg" in storage[user_id]
    if item_name != item.name.split('.')[0]:
        # Change storage
        storage[user_id][item.name] = storage[user_id][f"{item_name}.jpg"].copy()
        del storage[user_id][f"{item_name}.jpg"]

    save_storage(pseudo_s3_storage_file, storage)
    del storage
    pipeline._load_storage()
    
    item_query.update(item.dict(), synchronize_session=False)
    db.commit()

    # Get new item according to change
    item_query = db.query(models.Apparel).filter(models.Apparel.user_id==user_id, models.Apparel.name==item.name)

    return item_query.first()


#### DELETE ####
@router.delete("/{user_id}/{item_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(user_id: int, item_name: str, db: Session = Depends(get_db)):
    item_query = db.query(models.Apparel).filter(models.Apparel.user_id==user_id,
                                                models.Apparel.name==f"{item_name}.jpg")

    if item_query.first() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"item with name {item_name} was not found")

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
def outfits_recommend(user_id: int, outfit_recommend_option: schemas.OutfitsRecommendation, db: Session = Depends(get_db)):
    chosen = {}
    recommend_choices = []

    for cate in ["top", "bottom", "bag", "outerwear", "shoe"]:
        if isinstance(outfit_recommend_option.dict()[cate], str):
            img_name = outfit_recommend_option.dict()[cate]
            if img_name[-4:] != ".jpg":
                img_name = img_name + ".jpg"
            item = db.query(models.Apparel).filter(models.Apparel.user_id==user_id,
                                                models.Apparel.name==img_name).first()
            if item is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"item with name {img_name.split('.')[0]} was not found")
            chosen[cate] = img_name
        
        else:  # int
            if outfit_recommend_option.dict()[cate] == 1:  # Enable
                recommend_choices.append(cate)
    
    outputs = pipeline.outfit_recommend(chosen=chosen, recommend_choices=recommend_choices, db=db, user_id=user_id)
    return outputs