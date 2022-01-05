import torch
import torch.nn.functional as F
from mmcv.ops.point_sample import bilinear_grid_sample

# sample input and grid 
x = torch.randn(1, 4, 10, 10)
grid = 2*torch.rand(1, 8, 8, 2) - 1 # scale as (-1, 1)

# reference output
ref = F.grid_sample(x, grid, align_corners=False)
# substitute output
out = bilinear_grid_sample(x, grid, align_corners=False)
# almost the same
print(ref - out)

# Toy model including blinear_grid_sampler
class Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, grid):
        return bilinear_grid_sample(x, grid, align_corners=False)

torch.onnx.export(
    Sampler(),
    (x, grid),
    'bilinear_sampler.onnx',
    verbose=True,
    input_names=['input', 'grid'],
    output_names=['output']
)

# validate the converted onnx operation
import onnxruntime as ort

sess = ort.InferenceSession('bilinear_sampler.onnx')
outputs = sess.run(None, {'input': x.numpy(), 'grid': grid.numpy()})
out_onnx = outputs[0]

# almost the same
print(ref - out_onnx)