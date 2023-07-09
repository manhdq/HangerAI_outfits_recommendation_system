import os
import numpy as np
import cv2
import time
import copy
import yaml
import pickle5 as pickle
from PIL import Image

import torch

from Hash4AllFashion.utils.param import FashionExtractParam
from Hash4AllFashion.model import FashionNet


NO_WEIGHTED_HASH = 0
WEIGHTED_HASH_U = 1
WEIGHTED_HASH_I = 2
WEIGHTED_HASH_BOTH = 3


def get_features(feature_path):
    with open(feature_path, "rb") as handle:
        features = pickle.load(handle)

    return features


def get_inputs(chosen, from_database,
            recommend_choices,
            feature_path,
            score_type_selection="binary",
            feature_type_selection="visual"):
    inputs = [chosen]
    from_databases = [from_database]

    features = get_features(feature_path)

    def add_inputs(inputs, from_databases, choices):
        if len(choices.keys()) == 0:
            return inputs, from_databases

        picked_cate = list(choices.keys())[0]
        picked_choices = choices[picked_cate]
        del choices[picked_cate]

        new_inputs = []
        new_from_databases = []
        for input, from_database in zip(inputs, from_databases):
            cur_inputs = []
            cur_from_databases = []
            for picked_choice in picked_choices:
                input = copy.deepcopy(input)
                from_database = copy.deepcopy(from_database)

                input[picked_cate] = picked_choice
                from_database[picked_cate] = True

                cur_inputs.append(input)
                cur_from_databases.append(from_database)
            new_inputs.extend(cur_inputs)
            new_from_databases.extend(cur_from_databases)

        new_inputs, new_from_databases = add_inputs(new_inputs, new_from_databases, choices)
        return new_inputs, new_from_databases

    ## Get list of recommend_choices
    recommend_choices = copy.deepcopy(recommend_choices)
    for k, v in recommend_choices.items():
        if isinstance(v, str) and v=="all":
            recommend_choices[k] = list(features[k].keys())
        elif isinstance(v, (tuple, list)):  ##TODO: Remove redundancy
            if isinstance(v[0], int):
                assert len(v) == 2 
                recommend_choices[k] = list(features[k].keys())[v[0]:v[1]]
            else:
                assert isinstance(v[0], str)
                recommend_choices[k] = copy.deepcopy(v)

    inputs, from_databases = add_inputs(inputs, from_databases, recommend_choices)
    return inputs, from_databases, features


def get_net(config):
    """Get network"""
    net_param = config.net_param

    assert config.load_trained is not None

    # Dimension of latent codes
    net = FashionNet(net_param,
                    config.train_data_param.cate_selection)
    # Load model from pre-trained file
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    ##TODO: Lower ckpt file for faster loading
    state_dict = torch.load(config.load_trained, map_location=map_location)
    # load pre-trained model
    net.load_state_dict(state_dict)
    net.cuda(device=config.gpus[0])
    net.eval()  # Unable training
    return net


def compute_score(net, config, input, from_database, features, score_type_selection, scale=10.0):
    ilatents = [features[k][v] for k, v in input.items()]
    ilatents = np.stack(ilatents)
    ilatents = torch.from_numpy(ilatents).cuda(device=config.gpus[0])

    size = len(ilatents)
    indx, indy = np.triu_indices(size, k=1)
    # comb x D
    ##TODO: Rename
    x = ilatents[indx] * ilatents[indy]
    # Get score
    if config.net_param.hash_types == WEIGHTED_HASH_I:
        score_i = net.core(x).mean()
    else:
        ##TODO:
        raise

    ##TODO: Code for user score
    score = score_i * (scale * 2.0)
    return score.cpu().detach().item()


##TODO: Add select `score_type` and `feature_type` option
##TODO: Add extract features from new sample not exist in database
##TODO: Recommend composed outfit
##TODO: Later. Code batch computation for faster purpose
def outfit_recommend_from_chosen(
    cfg_path: str,
    chosen: dict,
    from_database: dict,
    recommend_choices: list,
    feature_path: str,
    score_type_selection: str = "binary",  ##TODO: Delete this for entire code
    feature_type_selection: str = "visual",
    num_recommends_per_choice: int = 5,
    num_recommends_for_composition: int = 3,
    get_composed_recommendation: bool = False,
    get_visual_image: bool = False,
    save_dir: str = "results",
    score_thres: float = 5.0,
):
    ##TODO: Load config and model before call API
    with open(cfg_path, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = FashionExtractParam(**kwargs)  ##TODO: Code for less reduncdancy

    net = get_net(config)

    start_time = time.time()
    outputs = {}
    for cate_choice in recommend_choices:
        scores = []
        inputs, from_databases, features = get_inputs(chosen, from_database,
                                                    {cate_choice: "all"},
                                                    feature_path=feature_path,
                                                    score_type_selection=score_type_selection,
                                                    feature_type_selection=feature_type_selection)
        
        for i, (input, cur_from_database) in enumerate(zip(inputs, from_databases)):
            score = compute_score(net, config, input, cur_from_database, features, score_type_selection)
            scores.append(score)
        target_arg = sorted(range(len(scores)), key=lambda k: -scores[k])

        cate_recommended = [inputs[i][cate_choice] for i in target_arg[:num_recommends_per_choice]]
        outputs[cate_choice] = cate_recommended

    if get_composed_recommendation:  # Recommend outfit from recommended above
        scores = []
        inputs, from_databases, features = get_inputs(chosen, from_database,
                                                    outputs,
                                                    feature_path=feature_path,
                                                    score_type_selection=score_type_selection,
                                                    feature_type_selection=feature_type_selection)
        
        for i, (input, cur_from_database) in enumerate(zip(inputs, from_databases)):
            score = compute_score(net, config, input, cur_from_database, features, score_type_selection)
            scores.append(score)
        target_arg = sorted(range(len(scores)), key=lambda k: -scores[k])
        recommeded_outfits = [inputs[i] for i in target_arg[:num_recommends_for_composition]]

        outputs["outfit_recommend"] = recommeded_outfits
    
    outputs["time"] = time.time() - start_time
    return outputs


if __name__ == "__main__":
    ##TODO: Config
    cfg_path = "configs/FHN_VSE_T3_visual.yaml"

    chosen = {"top": "104446629"}  ## Change this
    from_database = {"top": True}  ## Change this

    recommend_choices = ["bottom", "outerwear"]  ## Change this

    score_type_selection = "binary"
    assert score_type_selection in ["binary", "features"]
    feature_type_selection = "visual"
    assert feature_type_selection in ["visual", "semantic", "all"]

    feature_path = "data/val_binary_visual.pkl"

    num_recommends_per_choice = 5
    num_recommends_for_composition = 3
    get_composed_recommendation = True

    outputs = outfit_recommend_from_chosen(
        cfg_path=cfg_path,
        chosen=chosen,
        from_database=from_database,
        recommend_choices=recommend_choices,
        score_type_selection=score_type_selection,
        feature_type_selection=feature_type_selection,
        feature_path=feature_path,
        num_recommends_per_choice=num_recommends_per_choice,
        num_recommends_for_composition=num_recommends_for_composition,
        get_composed_recommendation=get_composed_recommendation
    )
    print(outputs)