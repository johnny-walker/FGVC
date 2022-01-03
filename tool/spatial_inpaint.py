from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import torch
import time


def spatial_inpaint(deepfill, mask, video_comp, nframe=1):
    start = time.time()
    if nframe == 1:
        keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
        with torch.no_grad():
            img_res = deepfill.forward(video_comp[:, :, :, keyFrameInd] * 255., mask[:, :, keyFrameInd]) / 255.
        video_comp[mask[:, :, keyFrameInd], :, keyFrameInd] = img_res[mask[:, :, keyFrameInd], :]
        mask[:, :, keyFrameInd] = False
        print('frame inpainting completed, consuming time:', time.time() - start)
    else:
        for indFrame in range(nframe):
            with torch.no_grad():
                img_res = deepfill.forward(video_comp[:, :, :, indFrame] * 255., mask[:, :, indFrame]) / 255.
            video_comp[mask[:, :, indFrame], :, indFrame] = img_res[mask[:, :, indFrame], :]
            mask[:, :, indFrame] = False
            print('frame inpainting completed, consuming time:', time.time() - start)     
    return mask, video_comp
