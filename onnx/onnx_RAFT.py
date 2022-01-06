import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from RAFT import utils
from RAFT import RAFT

import torch
import argparse
print(torch.__version__)

isONNX = False
if isONNX:
    import onnx
    import onnxruntime as onnxrun
else:
    from openvino.inference_engine import IECore

from PIL import Image
import numpy as np
import glob



DEVICE = 'cpu'
def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)) )
    model = model.module
    model.to(DEVICE)
    model.eval()

    return model

def convert_to_ONNX(args):
    RAFT_model = initialize_RAFT(args)
    
    # set the model to inference mode 
    RAFT_model.eval() 

    dummy_input  = (torch.randn(1, 3, 720, 1280, device=DEVICE), torch.randn(1, 3, 720, 1280, device=DEVICE))
    input_names  = ("image1", "image2")
    output_names = ("flow")

    torch.onnx.export(RAFT_model, 
                      dummy_input ,
                      args.onnx_name,
                      input_names = input_names, 
                      output_names = output_names,
                      opset_version = 11)

def check_model(args):
    model = onnx.load(args.onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_frames():
    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # debug, Obtains imgH, imgW and nFrame.
    #imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    #nFrame = len(filename_list)

    # Loads video.
    video = []
    for filename in sorted(filename_list):
        frame = Image.open(filename)
        frame = np.array(frame).astype(np.float32)
        frame = np.transpose(frame, (2, 0, 1))
        video.append(frame)

    video = np.stack(video, axis=0)
    return video, filename_list

def infer_flow_onnx(args):
    video, filename_list = read_frames()
    create_dir(os.path.join(args.outroot+'_flow', '_flo'))
    create_dir(os.path.join(args.outroot+'_flow', '_png'))

    # inference
    sess = onnxrun.InferenceSession(args.onnx_name)
    #inputs = sess.get_inputs()
    input_image1 = sess.get_inputs()[0].name
    input_image2 = sess.get_inputs()[1].name
    if sess is not None:
        for idx in range(len(video)-1):
            start = time.time()
            filename = os.path.split(filename_list[idx])[-1]
            filename = os.path.splitext(filename)[0]
            #filename =  + '%05d'%idx

            image1 = video[idx,   None]
            image2 = video[idx+1, None]
            result = sess.run( None, { input_image1: image1,
                                       input_image2: image2
                                     } )
            print(time.time()-start)
            flow = result[0]
            flow = flow.reshape((-1, flow.shape[2], flow.shape[3]))
            flow = np.transpose(flow, (1, 2, 0))
            flo_path = os.path.join(args.outroot+'_flow', '_flo', filename + '.flo')
            Image.fromarray(utils.flow_viz.flow_to_image(flow)).save(os.path.join(args.outroot+'_flow','_png', filename + '.png'))
            utils.frame_utils.writeFlow(flo_path, flow)

def infer_flow_openvino(args):
    video, filename_list = read_frames()
    create_dir(os.path.join(args.outroot+'_flow', '_flo'))
    create_dir(os.path.join(args.outroot+'_flow', '_png'))

    # inference
    ie = IECore()
    net = ie.read_network(model=args.onnx_name)
    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_blobs = []
    for item in net.input_info:
        input_blobs.append(item)

    if exec_net is not None:
        for idx in range(len(video)-1):
            filename = os.path.split(filename_list[idx])[-1]
            filename = os.path.splitext(filename)[0]
            #filename =  + '%05d'%idx

            image1 = video[idx,   None]
            image2 = video[idx+1, None]
            inputs = { input_blobs[0]: image1, input_blobs[1]: image2 }

            inferAsync = False
            start = time.time()
            if inferAsync:
                exec_net.requests[0].async_infer(inputs)
                request_status = exec_net.requests[0].wait()
                print(request_status)
                flow = exec_net.requests[0]
            else:
                result = exec_net.infer(inputs)
                flow = result[0]
            print(time.time()-start)
            
            flow = flow.reshape((-1, flow.shape[2], flow.shape[3]))
            flow = np.transpose(flow, (1, 2, 0))
            flo_path = os.path.join(args.outroot+'_flow', '_flo', filename + '.flo')
            Image.fromarray(utils.flow_viz.flow_to_image(flow)).save(os.path.join(args.outroot+'_flow','_png', filename + '.png'))
            utils.frame_utils.writeFlow(flo_path, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # RAFT
    parser.add_argument('--model', default='weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--onnx_name', default='weight_onnx/raft.onnx', help="saving path")
    parser.add_argument('--path', default='data/beach', help="soruce path")
    parser.add_argument('--outroot', default='data/beach', help="flo out path")
    args = parser.parse_args()

    if isONNX:
        if not os.path.exists(args.onnx_name) :
            # create folder
            folder = ''
            splits = os.path.split(args.onnx_name)
            for i in range(len(splits)-1):
                folder = os.path.join(folder, splits[i])
            create_dir(folder)

            # convert to onnx
            convert_to_ONNX(args)
            check_model(args)
        infer_flow_onnx(args)   # slow
    else:   # already convreted .onnx file successfully, load directly
        infer_flow_openvino(args)

