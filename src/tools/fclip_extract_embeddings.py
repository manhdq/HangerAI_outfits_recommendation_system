import os
import os.path as osp
import argparse

import numpy as np
import pandas as pd

from fashion_clip.fashion_clip import FashionCLIP


def parse_args():
    parser = argparse.ArgumentParser(description="Extract apparel feature vectors")
    parser.add_argument("-p", "--path", help="Csv file including each item with category pair")
    parser.add_argument("-d", "--img-dir", help="Path to image directory")
    parser.add_argument("-s", "--storage-path", help="Path to pickle file to store feature vectors")
    parser.add_argument("-b", "--batch-size", help="Batch size to run fashion-clip model", default=128)    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    fclip = FashionCLIP("fashion-clip")

    df_item_cate = pd.read_csv(args.path)
    img_ids = df_item_cate.id.tolist()
    print("Number of images:", len(img_ids))

    img_fpaths = [osp.join(args.img_dir, str(item_id)+".jpg") for item_id in img_ids]
    img_embeddings = fclip.encode_images(img_fpaths, args.batch_size)

    storage_path = args.storage_path
    if "txt" not in storage_path:
        storage_path = storage_path.split(".")[0] + ".txt"
        
    np.savetxt(storage_path, img_embeddings)
    print("Embedding successfully!!!")
