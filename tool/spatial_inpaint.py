from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import torch
import time


def spatial_inpaint(deepfill, mask, video_comp):
    start = time.time()
    keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
    with torch.no_grad():
        img_res = deepfill.forward(video_comp[:, :, :, keyFrameInd] * 255., mask[:, :, keyFrameInd]) / 255.
    video_comp[mask[:, :, keyFrameInd], :, keyFrameInd] = img_res[mask[:, :, keyFrameInd], :]
    mask[:, :, keyFrameInd] = False
    print('frame inpainting completed, consuming time:', time.time() - start)
    return mask, video_comp
