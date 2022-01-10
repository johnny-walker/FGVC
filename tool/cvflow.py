import cv2
import numpy as np
import glob
import os
import time

class CVFlowPredictor():
    def __init__(self):
        None
    
    def predict(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def write_viz(self, path, flow):
        def flow_viz(flow):
            # https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
            # https://answers.opencv.org/question/11006/what-is-the-carttopolar-angle-direction-in-opencv/
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            shape = (flow.shape[0], flow.shape[1], 3)
            hsv = np.zeros(shape, dtype=np.float32)
            hsv[..., 1] = 255                                                       # s channel
            hsv[..., 0] = angle * 180 / np.pi / 2                                   # h channel
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)   # v channel
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) # convert to BGR
        
        viz = flow_viz(flow)     
        cv2.imwrite(path, viz)

if __name__ == '__main__':
    def create_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def load_image(imfile):
        img = cv2.imread(imfile)
        return img

    cvflow = CVFlowPredictor()

    images = glob.glob(os.path.join('data/beach', '*.png')) + \
             glob.glob(os.path.join('data/beach', '*.jpg'))


    images = sorted(images)
    out_path = 'data/beach_cvFlow/'
    create_dir(out_path)

    for file1, file2 in zip(images[:-1], images[1:]):
        since = time.time()
        img1 = load_image(file1)
        img2 = load_image(file2)

        # calculate optical flow
        flow = cvflow.predict(img1, img2)
            
        filename = os.path.split(file1)[-1]
        name = os.path.splitext(filename)[0]
        output = os.path.join(out_path, "{:s}_flow.jpg".format(name))
        cvflow.write_viz(output, flow)
        print(output, time.time()-since)
        break
