#!/bin/sh
python tools/fclip_extract_embeddings.py \
  	 -p data/polyvore/item_cates.csv \
	 -d data/polyvore/images \
	 -s data/polyvore/embeddings/fclip_embeddings.txt
