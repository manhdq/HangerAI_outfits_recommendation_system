#!/bin/sh
python src/tools/fhn_extract_embeddings.py \
  	 -c configs/FHN_VOE_T3_fashion32.yaml \
	 -d data/polyvore/images \
	 -s data/polyvore/embeddings/fhn_embeddings.pkl
