# UD-NMT
This is the source code for [Lin H, Yao L, Yang B, et al. Towards User-Driven Neural Machine Translation. ACL 2021](https://aclanthology.org/2021.acl-long.310/), developed based on [OpenNMT-py](https://github.com/Waino/OpenNMT-py).

## Dataset
- Pretrain: WMT17 ZH-EN
- Finetune: UDT-Corpus ([download](https://drive.google.com/drive/folders/19XSb2a9gENh1xfZp3cSpKoap-yBwoMn3?usp=sharing))
  
We provide the pre-extracted cache keywords in UDT-Corpus. If you would like to extract your own cache keywords, put your `*.uid` and `meta.bin` file in `data/user/` (please refer to the README in UDT-Corpus for their format) and run the following command :
```
# Extract topic/context cache for train data with 8 processes. 
python make_cache.py -p 8 -m train -tl 25 -cl 35 -s data/user/mycache
# Extract similar/dissimilar user cache for test data with 8 processes.
python make_nb_cache.py -p 8 -m test -tl 25 -cl 35 -s data/user/mycache
```
Then the cache keywords (topic cache length=25, context cache length=35) will be generated in `data/user/mycache`.

## Setup
```
pip install -r requirements.opt.txt
```

## Preprocess
- After downloading the raw data, put them under `data/wmt17` and `data/user`, seperately.
- Run the following command to preprocess WMT17 and UDT-Corpus (including cache data):
  ```
  bash udnmt_tools/preprocess_pipeline.sh
  ```

## Pretrain
```
bash udnmt_tools/train_pipeline.sh
```
Training log will be saved under `onmt-runs/wmt17-xxx/`, models are under `onmt-runs/wmt17-xxx/models/`, and evaluation resuls are under `onmt-runs/wmt17-xxx/test/`.

## Finetune
- Modify `-CHECKPOINT` in  `finetune_pipeline.sh` (L16) to the best model got from pretraining.
- Run the following command：
  ```
  bash udnmt_tools/finetune_pipeline.sh
  ```
Training log will be saved under `onmt-runs/user-xxx/`, models are under `onmt-runs/user-xxx/models/`, and evaluation resuls are under `onmt-runs/user-xxx/test/`.

## Citation
If you would like to use this project or UDT-Corpus, please cite from the proceedings of ACL 2021:
```
@inproceedings{lin-etal-2021-towards,
	title = "Towards User-Driven Neural Machine Translation",
	author = "Lin, Huan  and
	Yao, Liang  and
	Yang, Baosong  and
	Liu, Dayiheng  and
	Zhang, Haibo  and
	Luo, Weihua  and
	Huang, Degen  and
	Su, Jinsong",
	booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
	month = aug,
	year = "2021",
	address = "Online",
	publisher = "Association for Computational Linguistics",
	doi = "10.18653/v1/2021.acl-long.310",
	pages = "4008--4018",
}
```