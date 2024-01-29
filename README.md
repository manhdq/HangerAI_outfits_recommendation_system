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
$ cd ${PATH tới repo này}
```

## Model Weights
Tải [pretrained weight](https://drive.google.com/file/d/19TDBoE4qQQg3JLXCbxnXtCCMUmZ7Rbn2/view?usp=drive_link)
```
$ unzip checkpoints.zip
```

## Configs
Mặc định code chạy dùng những file config sau đây. Có thể tạo mới hoặc dùng sẵn:

- File config recommend mẫu: [configs/polyvore_outfit_recommend.yaml](configs/polyvore_outfit_recommend.yaml)
- File config model pretrained mẫu: [configs/FHN_VOE_T3_fashion32.yaml](configs/FHN_VOE_T3_fashion32.yaml)

Nếu tạo mới thì cần chỉnh đường dẫn đến các file config này ở `line 10` file [src/api.py](src/api.py) và `line 147` file [app/outfit_recommend_app.py](app/outfit_recommend_app.py).

## Data

```
$ mkdir data
$ cd data
```

### Polyvore

 Tải [data polyvore](https://drive.google.com/file/d/1lVZ2Jj6oiL3aOzMN0sgcYUltCgcFMgu_/view?usp=drive_link) để vào trong thư mục data/ dùng để test
```
$ unzip polyvore.zip
```

### Custom data (nếu dùng data khác không dùng data Polyvore trên mới làm bước này)

- Tạo file csv chứa từng item id với category, ví dụ như ở dưới với tên file là **item_cates.csv** dùng cho ví dụ ở bước sau:

id | cate
--- | ---
29974 | top
27827 | top
28787 | bottom
29288 | shoe

- Extract feature vector từ ảnh:

Mẫu ở file [runs/preprocess/fhn_embedding.sh](runs/preprocess/fhn_embedding.sh) và [runs/preprocess/fclip_embedding.sh](runs/preprocess/fclip_embedding.sh)
```
$ chmod +x runs/preprocess/fhn_embedding.sh
$ sh runs/preprocess/fhn_embedding.sh
$ chmod +x runs/preprocess/fclip_embedding.sh
$ sh runs/preprocess/fclip_embedding.sh
```

## Run

### Api
Mở 1 tab chạy api rồi chờ một lúc cho đến khi chạy thành công:
```
$ sh runs/api/outfit_recommend.sh
```

Kết quả ví dụ sẽ có dạng như sau:

```
{
         “outfit_recommend”: [
                {
                        “top”: 183407434,
                        “bottom”: 179847434,
                        “shoe”: 164407434,
                        "outerwear" 164747434,
                        "bag": 189484039
                },
                {
                       “top”: 112456444,
                        “bottom”: 178747894,
                        “shoe”: 174648409,
                        "outerwear" 134407434,
                        "bag": 179407434
                },
                {
                        “top”: 133373934,
                        “bottom”: 123407434,
                        “shoe”: 108475434,
                        "outerwear": 199407434,
                        "bag": 146573039
                },
                …
        ],
	 "time": 2.8725521564483643
}
```

### App
Cùng lúc api chạy  thì mở 1 tab khác chạy:
```
$ sh runs/app/outfit_recommend.sh
```
