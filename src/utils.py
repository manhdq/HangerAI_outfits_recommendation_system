import base64
import time
from typing import List, Any
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from fashion_clip.fashion_clip import FashionCLIP


NO_WEIGHTED_HASH = 0
WEIGHTED_HASH_U = 1
WEIGHTED_HASH_I = 2
WEIGHTED_HASH_BOTH = 3


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


def base64_to_image(base64_string):
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Create an Image object from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image
