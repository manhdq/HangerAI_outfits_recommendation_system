import os.path as osp
from typing import List, Any
import copy
import time
from collections import defaultdict, namedtuple
import pickle
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from .base import *
from fashion_clip.fashion_clip import FashionCLIP
from icecream import ic


######## Global Variable ##########
NO_WEIGHTED_HASH = 0
WEIGHTED_HASH_U = 1
WEIGHTED_HASH_I = 2
WEIGHTED_HASH_BOTH = 3


######## Important Functions & Classes ##########
norm = lambda x: np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
name = lambda x: osp.basename(x).split('.')[0]

class Param(dict):
    def __init__(self, *args, **kwargs):
        super(Param, self).__init__(*args, **kwargs)
        self.__dict__ = self


def calculate_time(func):

    def timing(*args, **kwargs):
        t1 = time.time()
        outputs = func(*args, **kwargs)
        t2 = time.time()
        print(f"Time: {(t2-t1):.3f}s")

        return outputs

    return timing


def get_pretrained(net, path, device):
    # Load model from pre-trained file
    num_devices = torch.cuda.device_count()
    map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
    print(f"Load trained model from {path}")
    state_dict = torch.load(path, map_location=map_location)

    # load pre-trained model
    net.load_state_dict(state_dict)
    net.cuda(device=device)
    net.eval()  # Unable training
    return net    


######## Main Pipeline ##########

def clip_retrieve(
    model,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    text_embedding: np.ndarray = None,
    query: str = None,
    encode_images_func: Any = None,
    encode_text_func: Any = None,
    embeddings_file: str = None,
    save_embeddings: bool = False,
    batch_size: int = 64,
    verbose: bool = False
):
    """Search top images predicted by the model according to query

    Args:
        query (str): input query to search for fashion item
        image_paths (List[str]): list of image paths to retrieve from
        encode_images_func (Any): function to encode images
        encode_text_func (Any): function to encode texts
        embeddings_file (str): absolute or relative path to image embeddings file
        save_embeddings (bool): whether to save embeddings into a file
        batch_size (int): number of items in a batch to process

    Returns:
        List[str]: list of result image paths sorted by rank
        np.ndarray: text embedding for later use
    """
    if text_embedding is None:
        if verbose:
            print("Embedding text...")
        text_embedding = (
            encode_text_func if encode_text_func else model.encode_text
        )([query], batch_size)[0]
        text_embedding = text_embedding[np.newaxis, ...]

    if image_embeddings is None:
        image_embeddings = (
            encode_images_func
            if encode_images_func
            else model.encode_images
        )(image_paths, batch_size=batch_size)

        if embeddings_file is not None and save_embeddings:
            print("Save image embeddings...")
            np.savetxt(embeddings_file, image_embeddings)

    image_embeddings = image_embeddings / norm(image_embeddings)

    if verbose:
        print("Find matching fashion items...")
    cosine_sim = model._cosine_similarity(
        text_embedding, image_embeddings, normalize=True
    )
    inds = cosine_sim.argsort()[:, ::-1]
    inds = inds.squeeze().tolist()
    rank_image_paths = [image_paths[ind] for ind in inds]

    return rank_image_paths, text_embedding
    

class Pipeline:
    def __init__(self, param, device): 
        self.device = device
        self.outfit_recommend_option = defaultdict(list)

        ### Paths ###
        self.storage_path = param.path.storage_path
        self.image_embeddings = np.load(param.path.img_embedding_file)
        self.item_cate_map = pd.read_csv(param.path.item_info_file)

        ### Net ###
        self.hash_types = param.net.hash_types
        self.use_outfit_semantic = param.net.use_outfit_semantic

        self.clip = FashionCLIP("fashion-clip")
        if self.hash_types == WEIGHTED_HASH_I:
            core = CoreMat(param.net.dim, param.net.pairwise_weight),
        elif self.hash_types == WEIGHTED_HASH_BOTH:
            core = nn.ModuleList(
                [
                    CoreMat(param.net.dim, param.net.pairwise_weight),
                    CoreMat(param.net.dim, param.net.outfit_semantic_weight)
                ]
            )

        self.core = get_pretrained(core, param.path.core_trained, device)
        if self.use_outfit_semantic:
            encoder_o = TxtEncoder(param.net.outfit_semantic_dim, param.net)
            self.encoder_o = get_pretrained(encoder_o, param.path.encoder_o_trained, device)

        self.type_selection = {
            "latent": {
                "visual": "lci_v",
                "semantic": "lci_s",
            },
            "binary": {
                "visual": "bci_v",
                "semantic": "bci_s",
            },
        }[param.net.score_type_selection][param.net.feature_type_selection]

        ### Recommend ###
        self.cates = param.recommend.cates
        self.num_outfits = param.recommend.num_outfits
        self.top_k = param.recommend.top_k
        self.num_recommends_for_composition = param.recommend.num_recommends_for_composition
        self.get_composed_recommendation = param.recommend.get_composed_recommendation

        self._load_storage()

    def _load_storage(self):
        if osp.exists(self.storage_path):
            with open(self.storage_path, "rb") as file:
                loaded_data = pickle.load(file)
            self.storage = loaded_data
        else:
            self.storage = dict()

    def compute_score(self, input, olatent, scale=10.0):
        ilatents = [
            self.storage[v][self.type_selection]
            for _, v in input.items()
        ]

        ilatents = np.stack(ilatents)
        ilatents = torch.from_numpy(ilatents).cuda(device=self.device)

        size = len(ilatents)
        indx, indy = np.triu_indices(size, k=1)

        # comb x D
        pairwise = ilatents[indx] * ilatents[indy]
        semwise = 0
        if olatent is not None:
            semwise = ilatents * olatent
        
        score_o = 0
        # Get score
        if self.hash_types == WEIGHTED_HASH_I:
            score_i = self.core(pairwise).mean()
        elif self.hash_types == WEIGHTED_HASH_BOTH:
            score_i = self.core[0](pairwise).mean()
            score_o = self.core[1](semwise).mean()
        else:
            ##TODO:
            raise

        ##TODO: Code for user score
        score = (score_i + score_o) * (scale * 2.0)
        return score.cpu().detach().item()

    @staticmethod
    def add_inputs(self, inputs, choices):
        """Return pairs of garments to calculate compatibility score

        Args:
        inputs (dict): dictionary including which categories to choose from and which to recommend, usually in the type: [dict[cate: list[*.jpg"]]]
        choices (dict): one category with list of chosen items. "all" if choose all from database. In the form {"${cate}": list[*.jpg]}

        Returns:
        Pairs of items in the form list[{"cate1": *.jpg, "cate2": *.jpg, ..}, ...]
        """
        # inputs: [dict[cate: list[*.jpg"]]] ([chosen])
        # choices: {cate: list[*.jpg]} (recommended_choices)
        if len(choices.keys()) == 0:
            return inputs

        # Pick the first cate to merge
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

    def outfit_recommend_from_chosen(
        self,
        given_items: list[dict],
        recommend_choices: list,
        outfit_semantic = None    
    ):
        """Recommend outfit from database

        Args:
        given_items (dict): dictionary including which categories to choose from and which to recommend, usually in the type:
        {
            "${category}": Union[list[str], int]
        }
        """
        start_time = time.time()
        outputs = defaultdict(list)

        if outfit_semantic is not None:
            outfit_semantic = torch.from_numpy(outfit_semantic).cuda(device=self.device)
            outfit_semantic = self.encoder_o(outfit_semantic)

        # Recommend outfit from recommended above
        if self.get_composed_recommendation:  
            for item in given_items:
                scores = []
                inputs = self.add_inputs(self, [item], recommend_choices.copy())

                for input in inputs:
                    score = self.compute_score(
                        input,
                        outfit_semantic,
                    )
                    scores.append(score)
                target_arg = sorted(
                    range(len(scores)), key=lambda k: -scores[k]
                )
                recommended_outfits = [
                    inputs[i]
                    for i in target_arg[: self.num_recommends_for_composition]
                ][0]

                if recommended_outfits not in outputs["outfit_recommend"]:
                    outputs["outfit_recommend"].append(recommended_outfits)

                if len(outputs["outfit_recommend"]) >= self.num_outfits:
                    break

        outputs["time"] = time.time() - start_time
        return outputs

    @calculate_time
    def outfits_recommend_from_prompt(
        self,
        prompt: str,
    ):
        processed_text = prompt.lower()
        text_embedding = None

        # Retrieve top imgs from each cate
        for search_cate in self.cates:
            idxs = self.item_cate_map[self.item_cate_map["cate"] == search_cate].index.tolist()
            item_ids = self.item_cate_map.iloc[idxs, 1].tolist()
            embeddings = self.image_embeddings[idxs, :]
            found_image_ids, text_embedding = clip_retrieve(
                self.clip,
                image_embeddings=embeddings,
                image_paths=item_ids,
                text_embedding=text_embedding,
                query=processed_text
            )
            prompt_matched_image_ids = found_image_ids[:self.top_k]
            for image_id in prompt_matched_image_ids:
                # image_id = name(image_path) 
                self.outfit_recommend_option[search_cate].append(image_id)    

        # Compose outfits from retrieved items
        chosen = defaultdict(list)

        if not self.use_outfit_semantic:
            text_embedding = None

        # Recommend outfit not from just top items
        # of a particular category exclusively
        # but from top results retrieved of previous model
        first_cate = "top"

        # for image_id in self.outfit_recommend_option[first_cate]:
        given_items = [{first_cate: image_id} for image_id in self.outfit_recommend_option[first_cate]]
        recommend_choices = {
            k: v for k, v in self.outfit_recommend_option.items() if k != first_cate
        }
        outputs = self.outfit_recommend_from_chosen(
            given_items=given_items,
            recommend_choices=recommend_choices,
            outfit_semantic=text_embedding
        )

        self.outfit_recommend_option.clear()

        return outputs


def get_outfit_recommend(config_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = OmegaConf.load(config_path)
    return Pipeline(configs, device)


if __name__ == "__main__":
    model = Pipeline()
