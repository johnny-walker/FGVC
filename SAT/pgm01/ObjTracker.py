# -*- coding: utf-8 -*
import argparse
import os
import time
import glob
from tkinter.constants import N

import cv2
import numpy as np

import torch

from videoanalyst.config.config import cfg, specify_task
#from videoanalyst.engine.monitor.monitor_impl.utils import (labelcolormap, mask_colorize)
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
#from videoanalyst.utils.image import ImageFileVideoStream, ImageFileVideoWriter
#from videoanalyst.utils.visualization import VideoWriter

class VideoTracker:
    pipeline = None
    polygon_points = []
    videoFrames = []
    threshold = 0.5

    def __init__(self, args):
        self.initModel(args)

    def initModel(self, args):
        root_cfg = cfg
        root_cfg.merge_from_file(args.config)

        # resolve config
        root_cfg = root_cfg.test
        task, task_cfg = specify_task(root_cfg)
        task_cfg.freeze()

        # build model
        tracker_model = model_builder.build("track", task_cfg.tracker_model)
        tracker = pipeline_builder.build("track",
                                        task_cfg.tracker_pipeline,
                                        model=tracker_model)
        segmenter = model_builder.build('vos', task_cfg.segmenter)
        # build pipeline
        self.pipeline = pipeline_builder.build('vos',
                                        task_cfg.pipeline,
                                        segmenter=segmenter,
                                        tracker=tracker)
        dev = torch.device('cpu')
        self.pipeline.set_device(dev)

    def initData(self, videoFrames, points, threshold=0.01):
        if self.pipeline is not None:
            self.videoFrames = videoFrames
            self.polygon_points = points
            self.threshold = threshold
            return True
        return False

    def segmentFrames(self, callback):
        # init box and mask
        init_mask = None
        init_box = None

        first_frame = self.videoFrames[0]
        np_pts = np.array(self.polygon_points)
        init_box = cv2.boundingRect(np_pts)
        zero_mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
        init_mask = cv2.fillPoly(zero_mask, [np_pts], (1, ))
        self.pipeline.init(first_frame, init_box, init_mask)

        frame_idx = 0 
        for frame in self.videoFrames:
            time_a = time.time()
            score_map = self.pipeline.update(frame)
            mask = (score_map > self.threshold).astype(np.uint8) * 255
            time_cost = time.time() - time_a
            print("frame process, consuming time:", time_cost)
            three_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            callback(three_channel, frame_idx)
            frame_idx += 1

