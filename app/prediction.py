import os
import time
import copy
import numpy as np
import pickle5 as pickle
from collections import defaultdict
import snoop

import torch
import torch.nn.functional as F

from . import models
from Hash4AllFashion import utils
from Hash4AllFashion.utils import config as cfg
from Hash4AllFashion.model import FashionNet
from Hash4AllFashion.dataset.transforms import get_img_trans

##TODO: Change this
ID2CATE = {0: "top", 1: "bottom", 2: "bag", 3: "outerwear", 4: "shoe"}
CATE2ID = {"top": 0, "bottom": 1, "bag": 2, "outerwear": 3, "shoe": 4}

NO_WEIGHTED_HASH = 0
WEIGHTED_HASH_U = 1
WEIGHTED_HASH_I = 2
WEIGHTED_HASH_BOTH = 3


def get_net(config):
    """Get network."""
    # Get net param
    net_param = config.net_param

    assert config.load_trained is not None

    # Dimension of latent codes
    net = FashionNet(net_param, config.train_data_param.cate_selection)
    # Load model from pre-trained file
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    state_dict = torch.load(config.load_trained, map_location=map_location)
    # load pre-trained model
    net.load_state_dict(state_dict)
    net.cuda(device=config.gpus[0])
    net.eval()  # Unable training
    return net


class Pipeline:
    def __init__(self, config, storage_path):
        self.net = get_net(config)
        self.device = config.gpus[0]
        self.transforms = get_img_trans(
            "val", config.test_data_param.image_size
        )

        self.hash_types = config.net_param.hash_types

        self.type_selection = {
            "latent": {
                "visual": "lci_v",
                "semantic": "lci_s",
            },
            "binary": {
                "visual": "bci_v",
                "semantic": "bci_s",
            },
        }[config.score_type_selection][config.feature_type_selection]

        self.num_recommends_per_choice = config.num_recommends_per_choice
        self.num_recommends_for_composition = (
            config.num_recommends_for_composition
        )
        self.get_composed_recommendation = config.get_composed_recommendation

        self.storage_path = storage_path
        self._load_storage()

    def _load_storage(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as file:
                loaded_data = pickle.load(file)
            self.storage = loaded_data
        else:
            self.storage = dict()

    def extract_feats_from_one_sample(self, img_pil):
        img = img_pil.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        img = img.unsqueeze(0)
        img = utils.to_device(img, self.device)
        with torch.no_grad():
            (
                lcis_v,
                lcis_s,
                bcis_v,
                bcis_s,
                visual_logits,
            ) = self.net.extract_features(img)
            visual_fc = F.softmax(visual_logits, dim=1)

        ##TODO: Set semantic feats later
        lci_v = lcis_v[0].cpu().detach().numpy()
        lci_s = None
        bci_v = bcis_v[0].cpu().detach().numpy()
        bci_s = None
        cate_predict = np.argmax(visual_fc[0].cpu().detach().numpy())

        return lci_v, lci_s, bci_v, bci_s, ID2CATE[cate_predict]

    @staticmethod
    def add_inputs(self, inputs, choices):
        # inputs: chosen [dict[cate: list[*.jpg"]]]
        # choices: {cate: list[*.jpg]} (recommended_choices)
        if len(choices.keys()) == 0:
            return inputs

        picked_cate = list(choices.keys())[0]
        picked_choices = choices[picked_cate]
        del choices[picked_cate]  # delete category in choices

        new_inputs = []
        for input in inputs:
            cur_inputs = []
            for picked_choice in picked_choices:
                input = copy.deepcopy(input)

                input[picked_cate] = picked_choice

                cur_inputs.append(input)
            new_inputs.extend(cur_inputs)

        new_inputs = self.add_inputs(self, new_inputs, choices)
        return new_inputs

    def get_inputs(self, chosen, recommend_choices, db, user_id):
        """Return pairs of garments to calculate compatibility score

        Args:
        chosen (dict): dictionary including which categories to choose from and which to recommend, usually in the type:
        recommend_choices (dict): one category with list of chosen items. "all" if choose all from database. In the form {"${cate}": Union["all", list[*.jpg]}

        Returns:
        Pairs of items in the form list[dict{"cate1": [*.jpg], "cate2": [*.jpg]}]
        """
        inputs = [chosen]

        ## Get list of recommend_choices
        recommend_choices = copy.deepcopy(recommend_choices)
        for k, v in recommend_choices.items():
            if isinstance(v, str) and v == "all":
                items = (
                    db.query(models.Apparel)
                    .filter(
                        models.Apparel.user_id == user_id,
                        models.Apparel.category == k,
                    )
                    .all()
                )
                recommend_choices[k] = [item.name for item in items]
            elif isinstance(v, (tuple, list)):  ##TODO: Remove redundancy
                if isinstance(v[0], int):
                    assert len(v) == 2
                    items = (
                        db.query(models.Apparel)
                        .filter(
                            models.Apparel.user_id == user_id,
                            models.Apparel.category == k,
                        )
                        .all()
                    )
                    recommend_choices[k] = [item.name for item in items][
                        v[0] : v[1]
                    ]
                else:
                    assert isinstance(v[0], str)
                    recommend_choices[k] = copy.deepcopy(v)

        inputs = self.add_inputs(self, inputs, recommend_choices)
        return inputs

    def compute_score(self, net, input, user_id, scale=10.0):
        # print(self.storage.keys())
        # ilatents = [
        #     self.storage[user_id][v][self.type_selection]
        #     for _, v in input.items()
        # ]
        ilatents = [
            self.storage[user_id][v][self.type_selection]
            for _, v in input.items()
        ]
        ilatents = np.stack(ilatents)
        ilatents = torch.from_numpy(ilatents).cuda(device=self.device)

        size = len(ilatents)
        indx, indy = np.triu_indices(size, k=1)
        # comb x D
        x = ilatents[indx] * ilatents[indy]
        # Get score
        if self.hash_types == WEIGHTED_HASH_I:
            score_i = net.core(x).mean()
        else:
            ##TODO:
            raise

        ##TODO: Code for user score
        score = score_i * (scale * 2.0)
        return score.cpu().detach().item()

    def outfit_recommend(
        self, chosen: dict, recommend_choices: list, db, user_id
    ):
        """Recommend outfit from database

        Args:
        chosen (dict): dictionary including which categories to choose from and which to recommend, usually in the type:
        {
            "${category}": Union[list[str], int]
        }
        recommend_choices (list): list of categories to recomment
        """
        start_time = time.time()
        outputs = {}
        for cate_choice in recommend_choices:
            scores = []
            inputs = self.get_inputs(chosen, {cate_choice: "all"}, db, user_id)
            for i, input in enumerate(inputs):
                score = self.compute_score(self.net, input, user_id)
                scores.append(score)
            target_arg = sorted(range(len(scores)), key=lambda k: -scores[k])

            cate_recommended = [
                inputs[i][cate_choice]
                for i in target_arg[: self.num_recommends_per_choice]
            ]
            outputs[cate_choice] = cate_recommended

        if (
            self.get_composed_recommendation
        ):  # Recommend outfit from recommended above
            scores = []
            inputs = self.get_inputs(chosen, outputs, db, user_id)

            for i, input in enumerate(inputs):
                score = self.compute_score(self.net, input, user_id)
                scores.append(score)
            target_arg = sorted(range(len(scores)), key=lambda k: -scores[k])
            recommeded_outfits = [
                inputs[i]
                for i in target_arg[: self.num_recommends_for_composition]
            ]

            outputs["outfit_recommend"] = recommeded_outfits

        outputs["time"] = time.time() - start_time
        return outputs

    def outfit_recommend_from_chosen(
        self, given_items: list[dict], recommend_choices: list, db, user_id
    ):
        """Recommend outfit from database

        Args:
        chosen (dict): dictionary including which categories to choose from and which to recommend, usually in the type:
        {
            "${category}": Union[list[str], int]
        }
        """
        start_time = time.time()
        outputs = defaultdict(list)

        # for item in given_items:
        #     for cate_choice in recommend_choices:
        #         scores = []
        #         cate = cate_choice.keys()[0]
        #         inputs = self.add_inputs(self, [item], cate_choice)
        #         for i, input in enumerate(inputs):
        #             score = self.compute_score(self.net, input, user_id)
        #             scores.append(score)
        #         target_arg = sorted(range(len(scores)), key=lambda k: -scores[k])

        #         cate_recommended = [
        #             inputs[i][cate]
        #             for i in target_arg[: self.num_recommends_per_choice]
        #         ]
        #         outputs[cate] = cate_recommended

        if (
            self.get_composed_recommendation
        ):  # Recommend outfit from recommended above
            for item in given_items:
                scores = []
                inputs = self.get_inputs(item, recommend_choices, db, user_id)

                for i, input in enumerate(inputs):
                    score = self.compute_score(self.net, input, user_id)
                    scores.append(score)
                target_arg = sorted(
                    range(len(scores)), key=lambda k: -scores[k]
                )
                recommeded_outfits = [
                    inputs[i]
                    for i in target_arg[: self.num_recommends_for_composition]
                ][0]

                outputs["outfit_recommend"].append(recommeded_outfits)

        outputs["time"] = time.time() - start_time
        return outputs
