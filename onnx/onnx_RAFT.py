import os

from RAFT import utils
from RAFT import RAFT

import torch
import onnx
import argparse

print(torch.__version__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("cuda device not found, using cpu...")

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

    dummy_input  = (torch.randn(1, 3, 720, 1280, device=DEVICE), torch.randn(1, 3, 720, 1280,device=DEVICE))
    input_names  = ("image1", "image2", "iters")
    output_names = ("flow")

    torch.onnx.export(RAFT_model, 
                      dummy_input ,
                      args.onnx_name,
                      input_names = input_names, 
                      output_names = output_names,
                      opset_version = 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # RAFT
    parser.add_argument('--model', default='../weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('--onnx_name', default='weight_onnx/raft.onnx', help="saving path")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_name) :
        # create folder
        folder = None
        splits = os.path.split(args.onnx_name)
        for i in range(len(splits)-1):
            folder = os.path.join(folder, splits[i])
        os.makedirs(folder)

        # convert to onnx
        convert_to_ONNX(args)
