import os

import torch
import torchvision
print(torch.__version__)

import onnx
import onnxruntime as ort
import argparse
import numpy as np


DEVICE = 'cpu'
def convert_to_ONNX(args):
    dummy_input = torch.randn(10, 3, 224, 224, device=DEVICE)
    model = torchvision.models.alexnet(pretrained=True)

    # set the model to inference mode 
    model.eval() 

    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, 
                      dummy_input, 
                      args.onnx_name, 
                      verbose=True, 
                      input_names=input_names, 
                      output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_name', default='weight_onnx/alexnet.onnx', help="saving path")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_name) :
        # create folder
        folder = ''
        splits = os.path.split(args.onnx_name)
        for i in range(len(splits)-1):
            folder = os.path.join(folder, splits[i])
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # convert to onnx
        convert_to_ONNX(args)

    # Load the ONNX model
    model = onnx.load(args.onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    # inference
    ort_session = ort.InferenceSession(args.onnx_name)
    outputs = ort_session.run(None,
                              {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32) },
                              )
    print(outputs[0])