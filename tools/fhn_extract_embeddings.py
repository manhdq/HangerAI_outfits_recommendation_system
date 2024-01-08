# %%
import os
import os.path as osp
import yaml
import pickle
import importlib
import argparse

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import utils
from utils import config as cfg
from utils.param import FashionDeployParam
from model import fashionnet, get_net
from dataset.transforms import get_img_trans


CATE2ID = cfg.CateIdx
ID2CATE = {v: k for k, v in CATE2ID.items()}


class Pipeline:
    def __init__(self, config):
        self.net = get_net(config)
        self.transforms = get_img_trans("val", config.test_data_param.image_size)
        self.device = config.gpus[0]

    def extract_feats_from_one_sample(self, img):
        if self.transforms:
            img = self.transforms(image=img)["image"]
        img = img.unsqueeze(0)
        img = utils.to_device(img, self.device)
        with torch.no_grad():
            # (
            #     lcis_v,
            #     lcis_s,
            #     bcis_v,
            #     bcis_s,
            #     visual_logits,
            # ) = self.net.extract_features(img)
            feats = self.net.extract_features(img)

        visual_logits = feats["visual_fc"]
        lcis_v, bcis_v = feats["lcis_v"], feats["bcis_v"]

        visual_fc = F.softmax(visual_logits, dim=1)

        ##TODO: Set semantic feats later
        lci_v = lcis_v[0].cpu().detach().numpy()
        lci_s = None
        bci_v = bcis_v[0].cpu().detach().numpy()
        bci_s = None
        cate_predict = np.argmax(visual_fc[0].cpu().detach().numpy())

        return lci_v, lci_s, bci_v, bci_s, ID2CATE[cate_predict]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract apparel feature vectors")
    parser.add_argument("-c", "--cfg", help="Model config")
    parser.add_argument("-d", "--img-dir", help="Path to image directory")
    parser.add_argument("-s", "--storage-path", help="Path to pickle file to store feature vectors")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = FashionDeployParam(**kwargs)
    model = Pipeline(config)

    img_dir = args.img_dir
    img_fns = os.listdir(img_dir)

    apparel_feature_vecs = {}

    for img_fn in tqdm(img_fns):
        img_id = img_fn.split('.')[0]
        img_path = osp.join(img_dir, img_fn)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (
            lci_v,
            lci_s,
            bci_v,
            bci_s,
            cate_prediction,
        ) = model.extract_feats_from_one_sample(img)
        apparel_feature_vecs[img_id] = {
            "lci_v": lci_v,
            "lci_s": lci_s,
            "bci_v": bci_v,
            "bci_s": bci_s,
        }

    storage_path = args.storage_path
    if "pkl" not in storage_path:
        storage_path = storage_path.split(".")[0] + ".pkl"
    
    with open(storage_path, "wb") as f:
        pickle.dump(apparel_feature_vecs, f)
        f.close()

    print("Embedding successfully!!!")
