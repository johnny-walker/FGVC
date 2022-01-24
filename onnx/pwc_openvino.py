import argparse
import cv2
import os
import time
import glob
import numpy as np

from openvino.inference_engine import IECore
import pwc_utils

def reshape_input(net, pair):
    # Call reshape, but PWCNet seems not working
    x = np.array([pair])
    # padding for 64 alignment
    print('frame.shape:', x.shape)
    x_adapt, _ = pwc_utils.adapt_x(x)
    x_adapt = x_adapt.transpose((0, 4, 1, 2, 3))    # B2HWC --> BC2HW
    print('adapt.shape:', x_adapt.shape)
    print(f"Input shape: {net.input_info['x_tnsr'].tensor_desc.dims}")

    net.reshape({'x_tnsr': x_adapt.shape})
    print(f"Input shape (new): {net.input_info['x_tnsr'].tensor_desc.dims}")

def load_to_IE(model_xml, pair):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    #reshape_input(net, pair)
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")
    del net
    '''
    for item in exec_net.input_info:
        print('input:', item)
    for key in list(exec_net.outputs.keys()):
        print('output:', key)
    '''
    return exec_net

def read_frames(args):
    filename_list = glob.glob(os.path.join(args.input, '*.png')) + \
                    glob.glob(os.path.join(args.input, '*.jpg'))
    video = []
    for filename in sorted(filename_list):
        frame = cv2.imread(filename)
        frame = np.array(frame).astype(np.float32) / 255.   # normalize to range (0.0, 1.0)
        #frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        video.append(frame)
    #video = np.stack(video, axis=0) # 1CHW
    return video

def inference(args):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    video = read_frames(args)
    exec_net = load_to_IE(args.model, video[0])
    input_size = ( args.height, args.width )
    video_size = video[0].shape[:2]
    
    # resize to input, padding for 64 alignment
    print('original frame shape:', video[0].shape)
    image1, x_unpad_info = pwc_utils.resize_to_fit(video[0], input_size) # return HWC 
    print('adapt.shape:', image1.shape)
    image2 = None
    for idx in range(len(video)-1):
        image2, _ = pwc_utils.resize_to_fit( video[idx+1], input_size) # return HWC
        # Repackage input image pairs as np.ndarray
        x_adapt = np.array([(image1, image2)])  # --> B2HWC
        x_adapt = np.array([(video[idx], video[idx+1])]) 
        print (x_adapt.shape)
        x_adapt = np.array([[image1, image2]])  # --> B2HWC
        x_adapt = x_adapt.transpose((0, 4, 1, 2, 3))    # B2HWC --> BC2HW

        # inference
        start = time.time()
        y_hat = exec_net.infer({'x_tnsr':x_adapt})
        print(time.time()-start)
        image1 = image2

        # restore to orignal resolution, cut off the padding
        y_adapt = y_hat['pwcnet/flow_pred']
        y_adapt = y_adapt.transpose((0, 2, 3, 1))  # BCHW --> BHWC
        flow = np.squeeze(y_adapt, axis=0) #BHWC --> HWC
        flow = pwc_utils.unpad_and_upscale(flow, x_unpad_info, video_size)
        print (flow.shape)
        save_name = f'output/{idx:05d}.png'
        cv2.imwrite(save_name, pwc_utils.flow_to_img(flow))

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    model_desc = "location of the model XML file"
    input_desc = "location of the image input"

    parser.add_argument("--model", default='../models/model_ir_640x832/pwc_frozen.xml', help=model_desc)
    parser.add_argument("--input", default='D:/_PDR/_Shared/data/tennis', help=input_desc)
    parser.add_argument("--height", default=640, type=int, help='model input height')
    parser.add_argument("--width", default=832, type=int, help='model input width')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    main()