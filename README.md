# Giới thiệu

Project này được dùng để gợi ý trang phục

**Những loại đồ thời trang mà code có thể lấy ra được:**

- [x] Quần, áo, giày, túi xách, áo khoác
- [ ] Phụ kiện, trang sức

**Những tác vụ mà code có thể xử lý được:**

- [x] Gợi ý trang phục từ 1 mô tả đầu vào (prompt)
- [ ] Gợi ý trang phục từ ảnh 1 món đồ đầu vào
- [ ] Gợi ý trang phục từ ảnh 1 món đồ đầu vào cùng với mô tả cả bộ trang phục


# Sử dụng

## Setup
```
$ conda create -n outfit python=3.11
$ conda activate outfit
$ pip install -r requirements.txt
```

## Data

```
$ mkdir data
```

### Polyvore

Tải [weight](https://drive.google.com/file/d/19TDBoE4qQQg3JLXCbxnXtCCMUmZ7Rbn2/view?usp=drive_link) và [data polyvore](https://drive.google.com/file/d/1lVZ2Jj6oiL3aOzMN0sgcYUltCgcFMgu_/view?usp=drive_link) để vào trong thư mục data/ dùng để test
```
$ cd data
$ unzip checkpoints.zip
$ unzip polyvore.zip
```

### Custom data

- Tạo file csv chứa từng item id với category, ví dụ:
<p align="center">

id | cate
--- | ---
29974 | top
27827 | top
28787 | bottom
29288 | shoe

</p>

- Extract feature vector từ ảnh:
```
$ python tools/fhn_extract_embeddings.py \
  	 -c ${CONFIG_PATH} \
	 -d ${IMAGE_DIR} \
	 -s ${EMBEDDING_FILE}

$ python tools/fclip_extract_embeddings.py \
  	 -p ${CSV_ITEM_CATE} \
	 -d ${IMAGE_DIR} \
	 -s ${EMBEDDING_FILE}	 
```
VD:
```
$ python tools/fhn_extract_embeddings.py \
  	 -c configs/FHN_VOE_T3_fashion32.yaml \
	 -d data/polyvore/images \
	 -s data/polyvore/embeddings/fhn_embeddings.pkl

$ python tools/fclip_extract_embeddings.py \
  	 -p data/polyvore/item_cates.csv \
	 -d data/polyvore/images \
	 -s data/polyvore/fclip_embeddings.txt
```


## Configs
Mặc định code chạy dùng những file config sau đây. Có thể tạo mới hoặc dùng sẵn:

- File config recommend mẫu: [configs/polyvore_outfit_recommend.yaml](configs/polyvore_outfit_recommend.yaml)
- File config model pretrained mẫu: [configs/FHN_VOE_T3_fashion32.yaml](configs/FHN_VOE_T3_fashion32.yaml)

Sau đó chỉnh đường dẫn đến các file config ở `line 8` file [src/api.py](src/api.py) và `line 99` file [app/outfit_recommend_app.py](app/outfit_recommend_app.py)

## Run

### Api
```
$ ./runs/api/outfit_recommend.sh
```

### App
```
$ ./runs/app/outfit_recommend.sh
```
