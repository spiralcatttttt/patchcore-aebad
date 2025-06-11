# Patchcore implementation on AeBAD
This is an unofficial Patchcore implementation on dataset AeBAD<https://github.com/zhangzilongc/MMR> modified from official implementation <https://github.com/amazon-science/patchcore-inspection>. 

We add file aebad `/home/wangpeng/lsw/lsw/patchcore-inspection-main/src/patchcore/datasets/aebad.py` to use AeBAD_S dataset.

We also change the mvtec file to fix the bug that `self.split.TEST == DatasetSplit.TEST` is always false.

---
## Quick start

Enviroment setting can be seen in [README](official_README.md). We use `python==3.8, cuda=11.8, pytorch=2.0.0`.

Train on AeBAD_S after changing the dataset path.
```bash
bash train_AeBAD.sh
```


