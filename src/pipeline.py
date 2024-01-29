import os.path as osp
import copy
from collections import defaultdict, namedtuple
import pickle
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from .base import *
from .utils import *
from icecream import ic


class Pipeline:
    def __init__(self, param, device): 
        self.device = device
        self.outfit_recommend_option = defaultdict(list)

        ### Paths ###
        self._load_storage(param)

        ### Net ###
        self._load_net(param, device)

        ### Recommend ###
        self.cates = param.recommend.cates
        self.chosen_cates_first = param.recommend.chosen_cates_first
        self.num_outfits = param.recommend.num_outfits
        self.top_k = param.recommend.top_k
        self.num_recommends_for_composition = param.recommend.num_recommends_for_composition
        self.get_composed_recommendation = param.recommend.get_composed_recommendation

    def _load_storage(self, param):
        data_dir_path = param.path.data_dir_path
        embedding_dir_path = osp.join(data_dir_path, param.path.embedding_dir)

        # Load some files
        compose_embeddings_path = osp.join(embedding_dir_path, param.path.compose_embedding_file)
        if osp.exists(compose_embeddings_path):
            with open(compose_embeddings_path, "rb") as file:
                loaded_data = pickle.load(file)
            self.compose_embeddings = loaded_data
        else:
            self.compose_embeddings = dict()

        img_embeddings_path = osp.join(embedding_dir_path, param.path.retrieve_embedding_file)
        if "txt" in img_embeddings_path:
            self.retrieve_embeddings = np.loadtxt(img_embeddings_path)
        else:
            self.retrieve_embeddings = np.load(img_embeddings_path)            
        self.item_cate_map = pd.read_csv(osp.join(data_dir_path, param.path.item_cate_file))        

    def _load_net(self, param, device):
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
        
    def compute_score(self, input, olatent, scale=10.0):
        ilatents = [
            self.compose_embeddings[str(v)][self.type_selection]
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
                cate = list(item.keys())[0]
                scores = []
                recommend_choices = {
                    k: v for k, v in self.outfit_recommend_option.items() if k != cate
                }
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
                    inputs[indx]
                    for indx in target_arg[: self.num_recommends_for_composition]
                ]

                i = 0
                recommend_outfit = recommended_outfits[i]
                while recommend_outfit in outputs["outfit_recommend"]:
                    i += 1
                    recommend_outfit = recommended_outfits[i]
                    
                outputs["outfit_recommend"].append(recommend_outfit)

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
            item_ids = self.item_cate_map.iloc[idxs].id.tolist()
            embeddings = self.retrieve_embeddings[idxs, :]
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
        # first_cate = "top"
        # given_items = [{cate: image_id} for image_id in self.outfit_recommend_option[first_cate]]
        # recommend_choices = {
        #     k: v for k, v in self.outfit_recommend_option.items() if k != cate
        # }

        if isinstance(self.chosen_cates_first, list):
            given_items = [{cate: self.outfit_recommend_option[cate][0]} for cate in self.chosen_cates_first]
        elif isinstance(self.chosen_cates_first, str):
            cate = self.chosen_cates_first
            given_items = [{cate: image_id} for image_id in self.outfit_recommend_option[cate]]
        else:
            raise
        
        outputs = self.outfit_recommend_from_chosen(
            given_items=given_items,
            # recommend_choices=recommend_choices,
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
