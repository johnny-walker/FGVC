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
    print('frame inpainting completed, consuming time:', time.time - start)
    return mask, video_comp

# fill all frames
def spatial_inpaint_all(deepfill, mask, video_comp):
    with torch.no_grad():
        for index in range(len(video_comp)-1):
            start = time.time()
            img_res = deepfill.forward(video_comp[:, :, :, index] * 255., mask[:, :, index]) / 255.
            video_comp[mask[:, :, index], :, index] = img_res[mask[:, :, index], :]
            mask[:, :, index] = False
            print('frame {} inpainting completed, consuming time: {}'.format(index, time.time - start))
    return mask, video_comp