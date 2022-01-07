import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import imageio
from PIL import Image
import scipy.ndimage
import torchvision.transforms.functional as F
import time


from tool.get_flowNN import get_flowNN
from tool.spatial_inpaint import spatial_inpaint
from tool.frame_inpaint import DeepFillv1
import utils.region_fill as rf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from tool.cvflow import CVFlowPredictor
print("cuda device not found, using cpu...")

class VObjRemover():
    args = None
    def __init__(self, args):
        self.args = args

    def create_dir(self, dir):
        """Creates a directory if not exist.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

    def initialize_CVFlow(self):
        model = CVFlowPredictor()
        return model

    def initialize_RAFT(self):
        """Initializes the RAFT model.
        """
        model = torch.nn.DataParallel(RAFT(self.args))
        model.load_state_dict(torch.load(self.args.model, map_location=torch.device(DEVICE)) )
        model = model.module
        model.to(DEVICE)
        model.eval()

        return model

    def infer_flow(self, mode, filename, image1, image2, imgH, imgW, model):
        if DEVICE == 'cpu':
            frame1 = image1.reshape((-1, imgH, imgW)).cpu().numpy()
            frame1 = np.transpose(frame1, (1, 2, 0)).copy()
            frame2 = image2.reshape((-1, imgH, imgW)).cpu().numpy()
            frame2 = np.transpose(frame2, (1, 2, 0)).copy()
            flow = model.predict(frame1, frame2)
            #model.write_viz(os.path.join(self.args.outroot, 'flow', mode + '_png', filename + '.png'), flow)
        else:
            # original uters = 12
            _, flow = model(image1, image2, iters=int(self.args.iteration), test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow


    def calculate_flow(self, model, video):
        """Calculates optical flow.
        """
        start = time.time()
        nFrame, _, imgH, imgW = video.shape
        FlowF = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
        FlowB = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
        FlowNLF = np.empty(((imgH, imgW, 2, 3, 0)), dtype=np.float32)
        FlowNLB = np.empty(((imgH, imgW, 2, 3, 0)), dtype=np.float32)

        mode_list = ['forward', 'backward']

        for mode in mode_list:
            with torch.no_grad():
                for i in range(nFrame):
                    if mode == 'forward':
                        if i == nFrame - 1:
                            continue
                        # Flow i -> i + 1
                        print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
                        image1 = video[i, None]
                        image2 = video[i + 1, None]
                        flow = self.infer_flow(mode, '%05d'%i, image1, image2, imgH, imgW, model)
                        FlowF = np.concatenate((FlowF, flow[..., None]), axis=-1)
                    elif mode == 'backward':
                        if i == nFrame - 1:
                            continue
                        # Flow i + 1 -> i
                        print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
                        image1 = video[i + 1, None]
                        image2 = video[i, None]
                        flow = self.infer_flow(mode, '%05d'%i, image1, image2, imgH, imgW, model)
                        FlowB = np.concatenate((FlowB, flow[..., None]), axis=-1)
        
        print('Finish flow calculation. Consuming time:', time.time() - start)
        return FlowF, FlowB, FlowNLF, FlowNLB


    def complete_flow(self, corrFlow, flow_mask, mode):
        """Completes flow.
        """
        if mode not in ['forward', 'backward']:
            raise NotImplementedError

        sh = corrFlow.shape
        nFrame = sh[-1]

        compFlow = np.zeros(((sh)), dtype=np.float32)

        for i in range(nFrame):
            print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')

            flow = corrFlow[..., i]
            if mode == 'forward':
                flow_mask_img = flow_mask[:, :, i]
            elif mode == 'backward':
                flow_mask_img = flow_mask[:, :, i + 1]

            if mode == 'forward' or mode == 'backward':
                flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
                flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
                compFlow[:, :, :, i] = flow

        return compFlow


    def inference(self):
        begin = time.time()
        # Flow model.
        if DEVICE == 'cpu':
            RAFT_model = self.initialize_CVFlow()
        else:
            RAFT_model = self.initialize_RAFT()

        # Loads frames.
        filename_list = glob.glob(os.path.join(self.args.path, '*.png')) + \
                        glob.glob(os.path.join(self.args.path, '*.jpg'))

        # Obtains imgH, imgW and nFrame.
        imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
        nFrame = len(filename_list)

        # Loads video.
        video = []
        for filename in sorted(filename_list):
            video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())

        video = torch.stack(video, dim=0)
        video = video.to(DEVICE)

        # Calcutes the corrupted flow.
        corrFlowF, corrFlowB, _, _ = self.calculate_flow(RAFT_model, video)
        #print('\nFinish flow prediction.')

        start = time.time()
        # Makes sure video is in BGR (opencv) format.
        video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.


        # Loads masks.
        filename_list = glob.glob(os.path.join(self.args.path_mask, '*.png')) + \
                        glob.glob(os.path.join(self.args.path_mask, '*.jpg'))

        mask = []
        flow_mask = []
        for filename in sorted(filename_list):
            mask_img = np.array(Image.open(filename).convert('L'))
            mask.append(mask_img)

            # Dilate 15 pixel so that all known pixel is trustworthy
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
            # Close the small holes inside the foreground objects
            flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
            flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(bool)
            flow_mask.append(flow_mask_img)

        # mask indicating the missing region in the video.
        mask = np.stack(mask, -1).astype(bool)
        flow_mask = np.stack(flow_mask, -1).astype(bool)
        print('\nFinish filling mask holes. Consuming time:', time.time() - start)  

        # Completes the flow.
        videoFlowF = corrFlowF
        videoFlowB = corrFlowB
        start = time.time()
        videoFlowF = self.complete_flow(corrFlowF, flow_mask, 'forward')
        videoFlowB = self.complete_flow(corrFlowB, flow_mask, 'backward')
        print('\nFinish flow completion. Consuming time:', time.time() - start)

        iter = 0
        mask_tofill = mask
        video_comp = video

        # Image inpainting model.
        deepfill = DeepFillv1(pretrained_model=self.args.deepfill_model, image_shape=[imgH, imgW])

        # We iteratively complete the video.
        while(np.sum(mask_tofill) > 0):
            start = time.time()
            print('iteration:', iter)
            #self.create_dir(os.path.join(self.args.outroot, 'frame_comp_' + str(iter)))

            # Color propagation.
            video_comp, mask_tofill, _ = get_flowNN(self.args, video_comp, mask_tofill,     
                                                    videoFlowF, videoFlowB, None, None)

            print('\nFinish color propagation. Consuming time:', time.time() - start)
            for i in range(nFrame):
                mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
                img = video_comp[:, :, :, i] * 255
                # Green indicates the regions that are not filled yet.
                img[mask_tofill[:, :, i]] = [0, 255, 0]
                #cv2.imwrite(os.path.join(self.args.outroot, 'frame_comp_' + str(iter), '%05d.png'%i), img)


            start = time.time()
            # do color propagation at most n+1 times
            if self.args.inpainting or iter >= self.args.nProgagating:    
                mask_tofill, video_comp = spatial_inpaint(deepfill, mask_tofill, video_comp, nFrame)
                break
            else:
                mask_tofill, video_comp = spatial_inpaint(deepfill, mask_tofill, video_comp)
                iter += 1

        print('Total consuming time:', time.time() - begin)    
        finalname = os.path.split(self.args.path)[-1]
 
        self.create_dir(os.path.join(self.args.outroot, 'frame_comp_' + 'final'))
        video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]

        # save mp4
        filename = os.path.join(self.args.outroot, 'frame_comp_' + 'final', finalname+'.mp4')
        imageio.mimwrite(filename, video_comp_, fps=15, quality=8, macro_block_size=1)
        print('saved file:', filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # following args are required
    # video completion
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='data/beach', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='data/beach_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='data/vc', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--deepfill_model', default='weight/imagenet_deepfill.pth', help="restore checkpoint")

    # extra optional args
    parser.add_argument('--inpainting', action='store_true', help='all the remaining unknown pixels apply inpainting')
    parser.add_argument('--nProgagating', default=2, help="do color progagating at most n+1 time")

    args = parser.parse_args()

    vObjRemover = VObjRemover(args)
    vObjRemover.inference()
