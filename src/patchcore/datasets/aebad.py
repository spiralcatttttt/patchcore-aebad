import os
from enum import Enum

import PIL
from glob import glob
import torch
from torchvision import transforms
from .mvtec import MVTecDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AeBADDataset(MVTecDataset):
    """
    PyTorch Dataset for AeBAD.
    """

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")  # 取消注释
            
            # 确保只处理目录
            anomaly_types = [i for i in os.listdir(classpath) 
                            if os.path.isdir(os.path.join(classpath, i))]
            
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                imgpaths_per_class[classname][anomaly] = []
                maskpaths_per_class[classname][anomaly] = []  # 初始化掩码路径列表
                
                # 处理所有子类型
                sub_types = [i for i in os.listdir(anomaly_path)
                            if os.path.isdir(os.path.join(anomaly_path, i))]
                
                for sub_type in sub_types:
                    sub_path = os.path.join(anomaly_path, sub_type)
                    
                    # 加载图像路径
                    img_files = glob(os.path.join(sub_path, "*.png"))
                    imgpaths_per_class[classname][anomaly].extend(img_files)
                    
                    # 处理掩码路径（仅测试集且非正常类别）
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        mask_sub_path = os.path.join(maskpath, anomaly, sub_type)
                        
                        # 确保掩码目录存在
                        if os.path.exists(mask_sub_path):
                            mask_files = [os.path.join(mask_sub_path, os.path.basename(f)) 
                                        for f in img_files]
                            maskpaths_per_class[classname][anomaly].extend(mask_files)
                        else:
                            # 处理缺失掩码的情况
                            maskpaths_per_class[classname][anomaly].extend([None] * len(img_files))
                    else:
                        # 非测试集或正常类别，无掩码
                        maskpaths_per_class[classname][anomaly].extend([None] * len(img_files))

        # 展开数据结构
        data_to_iterate = []
        for classname in imgpaths_per_class:
            for anomaly in imgpaths_per_class[classname]:
                for i, img_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    mask_path = maskpaths_per_class[classname][anomaly][i]
                    data_to_iterate.append([classname, anomaly, img_path, mask_path])

        return imgpaths_per_class, data_to_iterate