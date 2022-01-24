import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from RAFT import utils

import torch
import argparse
print(torch.__version__)
from openvino.inference_engine import IECore

from PIL import Image
import numpy as np
import glob


DEVICE = 'cpu'
def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_frames():
    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Loads video.
    frames = []
    for filename in sorted(filename_list):
        frame = Image.open(filename)
        frame = np.array(frame).astype(np.float32)
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    frames = np.stack(frames, axis=0)
    return frames, filename_list

def infer_flow_openvino(args):
    frames, filename_list = read_frames()
    create_dir(os.path.join(args.outroot+'_flow', '_flo'))
    create_dir(os.path.join(args.outroot+'_flow', '_png'))

    # inference
    ie = IECore()
    net = ie.read_network(model=args.onnx_model)
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_blobs = []
    for item in net.input_info:
        input_blobs.append(item)
    net_outputs = list(net.outputs.keys())

    if exec_net is not None:
        for idx in range(len(frames)-1):
            filename = os.path.split(filename_list[idx])[-1]
            filename = os.path.splitext(filename)[0]
            #filename =  + '%05d'%idx

            inputs = { input_blobs[0]: [(frames[idx], frames[idx+1])] }

            start = time.time()
            outputs = exec_net.infer(inputs)
            flow = outputs[net_outputs[0]]
            print(time.time()-start)
            
            flow = flow.reshape((-1, flow.shape[2], flow.shape[3]))
            flow = np.transpose(flow, (1, 2, 0))
            flo_path = os.path.join(args.outroot+'_flow', '_flo', filename + '.flo')
            Image.fromarray(utils.flow_viz.flow_to_image(flow)).save(os.path.join(args.outroot+'_flow','_png', filename + '.png'))
            utils.frame_utils.writeFlow(flo_path, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='data/beach', help="soruce path")
    parser.add_argument('--outroot', default='data/beach', help="flo out path")
    args = parser.parse_args()

    # already convreted .onnx file successfully, load directly
    infer_flow_openvino(args)

