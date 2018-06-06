# coding＝utf-8
import sys
import numpy as np
import tifffile as tiff
import pydensecrf.densecrf as dcrf
import tensorflow as tf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import skimage.io as io
import cv2
from numpy import random
def crf(image, softmax):
unary = softmax_to_unary(softmax)  # 转为一元.
unary = np.ascontiguousarray(unary)
d = dcrf.DenseCRF2D(image.shape[0], image.shape[1], 2)
d.setUnaryEnergy(unary)
feats = create_pairwise_gaussian(sdims=(4, 4), shape=image.shape[:2])    # (5,5)  #(10,10)
    
    d.addPairwiseEnergy(feats, compat=3,
                        
                        kernel=dcrf.DIAG_KERNEL,
                        
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
# 创建与颜色相关的图像特征——因为我们从卷积神经网络中得到的分割结果非常粗糙,使用局部的颜色特征来改善分割结果
feats = create_pairwise_bilateral(sdims=(15, 15), schan=(10, 10, 10), img=image, chdim=2)
d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,     normalization=dcrf.NORMALIZE_SYMMETRIC)   # d是一个惩罚项好像,是随机分类的指标.
q = d.inference(2)
res = np.argmax(q, axis=0).reshape((image.shape[0], image.shape[1]))
res = res.astype(np.uint8)
return res
