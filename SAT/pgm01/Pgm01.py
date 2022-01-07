import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import tkinter as tk
#from tkinter import messagebox
from tkinter import filedialog
import cv2
import numpy as np
import threading
#import time
import glob

# import own modules
from ProgramBase import PgmBase
from ThreadBase import ThreadClass
from Utils import Pixels
from ObjTracker import VideoTracker
from ObjRemover import VObjRemover

class VideoInpaint(PgmBase):
    videoObject = None
    videofile = None
    curFrame = None
    curMask = None
    frameIndex = 0
    videoFrames = []    
    maskFrames = [] 
    removedFrames = []   

    drawRectangle = True
    isSelection = False
    selectionPts = []               # at most 4 points, original user selection points
    circles = []                    # at most 4 circles, keep the drawn ids
    idRectangle = -1

    blending = True
    alpha = 1.0
    drawRemovedFrame = False

    isBrushing = False
    isBrushAdd = True
    brushSize = 20
    autoBlending = False

    def __init__(self, root, width=800, height=600, args=[]):
        super().__init__(root, width, height)
        self.title = 'Frame Viewer'
        self.root.title(self.title)
        self.args = args

        # initi thread for video playback
        self.thread = None
        self.threadEventPlayback = threading.Event()

        self.pixels = Pixels()
        self.changeBtnStyle('blend', self.blending)

    # --- handle video and generate mask ---
    def openVideo(self):
        self.videofile = self.args.video
        if self.videofile:
            self.showMessage("Open video : {0:s}".format(self.videofile))
            self.threadEventPlayback.set()
            self.thread = ThreadClass(1, "Frame Reading Thread", self, self.readVideoFrameThreadFun)
            self.threadEventPlayback.clear() 
            self.thread.start()

    # thread function
    def readVideoFrameThreadFun(self):
        def _initVideoFrame():
            self.videoObject = cv2.VideoCapture(self.videofile)
            if self.videoObject.isOpened():
                ret, frame = self.videoObject.read()
                if ret:
                    self.curFrame = self.resize(frame)
                    self.videoFrames.append(self.curFrame)
                    self.drawFrame()  # draw current frame
                return ret
            return False

        def _readFrame():
            ret, frame = self.videoObject.read()
            if ret:
                frame = self.resize(frame)
                self.videoFrames.append(frame)
            else:
                return False # break
            return True   # continue reading

        ret = _initVideoFrame()
        if ret:
            while ret: 
                ret = _readFrame()
                if self.threadEventPlayback.wait(0):
                    break
            self.videoObject.release()
            self.threadEventPlayback.clear()
        self.isSelection = True
        print('thread stopped, all frames in memery...')
    
    def doSegmentation(self):
        if self.isSelection :
            self.destroyDrawObjects()
            self.threadEventPlayback.set()
            self.thread = ThreadClass(2, "Frame Segmentation Thread", self, self.segmentFramesThreadFun)
            self.threadEventPlayback.clear() 
            self.thread.start()
        self.isSelection = False

    # thread function
    def segmentFramesThreadFun(self):
        def _callback(mask, index):
            print('process frame:', index)
            self.maskFrames.append(mask)
            self.frameIndex = index
            self.refreshFrame() 

        self.sat = VideoTracker(self.args)
        if self.sat.initData(self.videoFrames, self.selectionPts, self.args.threshold):
            self.sat.segmentFrames(_callback)
        print('thread stopped, sagmentation done...')


    def doObjRemoval(self):
        if not self.isSelection and len(self.videoFrames)>1 and len(self.maskFrames)>1 :
            self.threadEventPlayback.set()
            self.thread = ThreadClass(3, "Object Remover Thread", self, self.removeObjectThreadFun)
            self.threadEventPlayback.clear() 
            self.thread.start()

    def removeObjectThreadFun(self):
        def _callback(updateFrame, index):
            print('process frame:', index)
            self.removedFrames.append(updateFrame)
            self.refreshFrame() 

        self.fgvc = VObjRemover(self.args)
        if self.fgvc is not None:
            self.fgvc.inference(self.videoFrames, self.maskFrames, _callback)
        print('thread stopped, object removing done...')

    # --- handle frames ---
    def loadData(self):
        # load data frames
        filename_list = glob.glob(os.path.join(self.args.path, '*.png')) + \
                        glob.glob(os.path.join(self.args.path, '*.jpg'))

        firstFrame = True
        for filename in sorted(filename_list):
            frame = self.loadImage(filename)
            self.videoFrames.append(frame)
            if firstFrame:
                self.curFrame = frame.copy()
                firstFrame = False

        # load mask
        filename_list = glob.glob(os.path.join(self.args.mask, '*.png')) + \
                        glob.glob(os.path.join(self.args.mask, '*.jpg'))

        firstMask = True
        for filename in sorted(filename_list):
            frame_mask = self.loadImage(filename)
            self.maskFrames.append(frame_mask)
            if firstMask:
                self.curMask = frame_mask.copy()
                firstMask = False
                self.drawFrame()  

    ### --- overrite button handlers ---
    def onPrev(self):
        if self.frameIndex > 0 :
            self.frameIndex -= 1
            self.refreshFrame() 
            self.showMessage("Navigate to frame - {0}".format(self.frameIndex))
    
    def onNext(self):
        if self.frameIndex < len(self.videoFrames)-1 :
            self.frameIndex += 1
            self.refreshFrame()  
            self.showMessage("Navigate to frame - {0}".format(self.frameIndex))

    def updateBrushStyle(self):
        self.changeBtnStyle('brush', self.isBrushing)
        self.changeBtnStyle('brush_add', self.isBrushing and self.isBrushAdd)
        self.changeBtnStyle('brush_erase', self.isBrushing and not self.isBrushAdd)
    
    def autoEnableBlending(self):
        # auto enable blending, as brush is used to change mask
        if self.isBrushing and not self.blending:
            self.onBlend()
            self.autoBlending = True

        # undo blending if it is enabled by auto blending
        if not self.isBrushing and self.blending and self.autoBlending:
            self.autoBlending = False
            self.onBlend() 

    def onBrush(self):
        self.isSelection = False
        self.isBrushing = not self.isBrushing
        self.updateBrushStyle()
        
        # update cursor
        style = "circle" if self.isBrushing else "arrow"
        self.changeCursor(style)

        self.autoEnableBlending()

    def onBrushAdd(self):
        self.isBrushAdd = True
        self.updateBrushStyle()
        self.autoEnableBlending()

    def onBrushErase(self):
        self.isBrushAdd = False
        self.updateBrushStyle()
        self.autoEnableBlending()

    def onBlend(self):
        self.autoBlending = False
        self.isSelection = False
        self.blending = not self.blending
        self.changeBtnStyle('blend', self.blending)
        self.drawFrame()  

    def onReset(self):
        # reset selection
        self.showMessage("Selection reset")
        self.selectionPts = []          
        self.destroyDrawObjects()

    def onSave(self):
        # save frames
        if len(self.args.path) > 0:
            if not os.path.exists(self.args.path):
                os.makedirs(self.args.path)
            idx = 0
            for frame in self.videoFrames:
                path = os.path.join(self.args.path, "{:06d}.jpg".format(idx))
                cv2.imwrite(path, frame)
                idx += 1 
        # save masks
        if len(self.args.mask) > 0:
            if not os.path.exists(self.args.mask):
                os.makedirs(self.args.mask)
            idx = 0
            for mask in self.maskFrames:
                path = os.path.join(self.args.mask, "{:06d}_mask.jpg".format(idx))
                cv2.imwrite(path, mask)
                idx += 1
        print('saving done...')

    def saveMP4(self):
        filename = os.path.join(args.path, 'video.mp4')
        _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, _fourcc, 30.0, (self.videoFrames[0].shape[1],self.videoFrames[0].shape[0]))
        for frame in self.videoFrames:
            out.write(frame)
        print('saved file:', filename)

    ### --- event handlers ---
    def onKey(self, event):
        if event.char == event.keysym or len(event.char) == 1:
            if event.keysym in ['Left', 'Right', 'Up', 'Down'] :
                self.onKeyArrors(event.keysym)
            elif event.char in ['<',',']:
                self.onPrev()     
            elif event.char in ['>', '.']:
                self.onNext()     
            elif event.char in ['+', '=']:
                self.onBrushAdd()     
            elif event.char in ['-', '_']:
                self.onBrushErase() 
            elif event.char in ['s', 'S']:
                self.onSave()     
            elif event.char in ['v', 'V']:
                self.saveMP4() 
            elif event.keysym == 'space':
                self.doSegmentation()    
            elif event.char in ['r', 'R']:
                self.doObjRemoval()    
            elif event.keysym == 'Escape':
                self.onExit()
        else:
            print (event.keysym)
    
    def onKeyArrors(self, keysym):
        if keysym == 'Left' :
            self.frameIndex = max(self.frameIndex-5, 0)
        elif keysym == 'Right' :
            self.frameIndex = min(self.frameIndex+5, len(self.videoFrames)-1)
        elif keysym == 'Up' :
            self.frameIndex = 0
        elif keysym == 'Down' :
            self.frameIndex = len(self.videoFrames)-1
        self.refreshFrame() 
        self.showMessage("Navigate to frame - {0}".format(self.frameIndex))

    def onExit(self):
        self.threadEventPlayback.set()
        self.thread = None
        if self.thread is not None and self.srcVideoObj is not None:
            self.srcVideoObj.release()
            self.srcVideoObj = None
            self.thread = None
        self.root.destroy()

    ### --- update frame content---
    def refreshFrame(self):
        self.curFrame = self.videoFrames[self.frameIndex].copy()
        if len(self.maskFrames) > 0 and self.frameIndex < len(self.maskFrames):
            self.curMask = self.maskFrames[self.frameIndex].copy()
        self.drawFrame()

    def drawFrame(self):
        if self.drawRemovedFrame and self.frameIndex < len(self.removedFrames):
            self.updateImage(self.removedFrames[self.frameIndex])
            return

        # draw on canvas
        if len(self.maskFrames) > 0 and self.blending:
            if self.mouseLeftDown:
                self.curFrame = self.videoFrames[self.frameIndex].copy()
            beta = ( 1.0 - self.args.alpha )
            cv2.addWeighted( self.curMask, self.args.alpha, self.curFrame, beta, 0.0, self.curFrame)

        self.updateImage(self.curFrame)
        if self.drawRectangle:
            self.drawRect(self.selectionPts)
 
    ### --- canvas drawing funcs ---
    def destroyDrawObjects(self):
        if self.idRectangle:
            self.canvas.delete(self.idRectangle)
            self.idRectangle = -1
        for id in self.circles:  
            self.canvas.delete(id)
        self.circles = []

    def drawRect(self, pts):   
        if len(pts) == 4 and self.isSelection : 
            color = 'red' #if self.isSelection else 'purple'
            dash = (8, 2) #if self.isSelection else (5, 2)
            self.canvas.delete(self.idRectangle)
            self.idRectangle = self.canvas.create_line( int(pts[0][0]+self.imageStartPos[0]), int(pts[0][1]+self.imageStartPos[1]), 
                                                        int(pts[1][0]+self.imageStartPos[0]), int(pts[1][1]+self.imageStartPos[1]),
                                                        int(pts[2][0]+self.imageStartPos[0]), int(pts[2][1]+self.imageStartPos[1]),
                                                        int(pts[3][0]+self.imageStartPos[0]), int(pts[3][1]+self.imageStartPos[1]),
                                                        int(pts[0][0]+self.imageStartPos[0]), int(pts[0][1]+self.imageStartPos[1]),
                                                        fill=color,
                                                        width=2,
                                                        dash=dash)
        else:
            self.destroyDrawObjects()

        # update circles
        for id in self.circles:  
            self.canvas.delete(id)
        if self.isSelection:
            self.circles = []
            for p in pts:
                id = self.create_circle(p[0]+self.imageStartPos[0], p[1]+self.imageStartPos[1], 2, self.canvas)
                self.circles.append(id)
    
    # (x,y): center, r: radius
    def create_circle(self, x, y, r, canvas): 
        x0, y0 = x-r, y-r
        x1, y1 = x+r, y+r
        return canvas.create_oval(x0, y0, x1, y1, fill="orange", outline='orange', width=3)

    # selection to add control points
    def updateCloudPoints(self, mousePt):
        def _replaceNearestSelectionPt(mousePt):
            mindist = 1000000
            indexMin = index = 0
            for pt in self.selectionPts:
                dist = self.pixels.norm2Distance(mousePt, pt)
                if dist < mindist:
                    mindist = dist
                    indexMin = index
                index += 1
            self.selectionPts[indexMin] = mousePt

        if len(self.selectionPts) < 4:
            self.selectionPts.append(mousePt)
        else:
            _replaceNearestSelectionPt(mousePt)
        
        if self.drawRectangle:
            self.drawRect(self.selectionPts)

    def mouseLClick(self, event):
        if self.isSelection:
            if self.hitTestImageRect(event, self.imageClickPos):
                print('({}, {})'.format(self.imageClickPos[0], self.imageClickPos[1]))
                self.updateCloudPoints(self.imageClickPos)

    def mouseLRelease(self, event):
        super().mouseLRelease(event) 
        if len(self.maskFrames) > 0 and self.frameIndex < len(self.maskFrames):
            self.maskFrames[self.frameIndex] = self.curMask.copy()
            self.refreshFrame()

    def mouseMove(self, event):
        super().mouseMove(event) 
        if not self.isSelection and self.isBrushing and self.mouseLeftDown:
            #print('painting', self.imgPosX, self.imgPosY)
            color = (64, 128, 64)
            cv2.circle(self.curFrame, (self.imgPosX, self.imgPosY), self.brushSize, color, -1)
            maskColor = (255, 255, 255) if self.isBrushAdd else  (0, 0, 0)
            cv2.circle(self.curMask, (self.imgPosX, self.imgPosY), self.brushSize, maskColor, -1)
            self.drawFrame()
    
    def mouseWheel(self, event):
        if self.isBrushing:
            self.brushSize += event.delta
            self.brushSize = max(self.brushSize, 3)
            self.showMessage("Brush size = {0:03d}".format(self.brushSize))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='video/sat/cheetah', help="input data folder")
    parser.add_argument('--mask', default='video/sat/cheetah_mask', help="input data mask folder")
    parser.add_argument('--video', default='video/cheetah.mp4', help="input video")
    parser.add_argument('--scaler', default=1.0, help="resize video frames") 
    parser.add_argument('--threshold', default=0.005, help="threshold to create binary mask") 

    # video completion
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    #parser.add_argument('--path', default='data/beach', help="dataset for evaluation")
    #parser.add_argument('--path_mask', default='data/beach_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='result/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--deepfill_model', default='weight/imagenet_deepfill.pth', help="restore checkpoint")
    
    args = parser.parse_args()


    # process first file to get shape
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))
    
    if len(filename_list) > 0:
        img = cv2.imread(filename_list[0])
        height, width = img.shape[0], img.shape[1]
        img = None
        program = VideoInpaint(tk.Tk(), width, height, args)
        program.loadData()
        program.run()
    else:   # process video
        videoObject = cv2.VideoCapture(args.video)
        if videoObject.isOpened():
            ret, frame = videoObject.read()
            if ret:
                height, width = int(frame.shape[0]*args.scaler), int(frame.shape[1]*args.scaler)
                program = VideoInpaint(tk.Tk(), width, height, args)
                program.openVideo()
                program.run()
