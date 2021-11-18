from torch.profiler import profile, record_function, ProfilerActivity
from utils import count_params, import_class
import torch

Model = import_class('model.ctrgcn.Model')
ctrgcn_args = {
    'in_channels': 3,
    'num_class': 60,
    'num_point': 25,
    'num_person': 2,
    'graph': 'graph.ntu_rgb_d.AdjMatrixGraph'
}

the_model = Model(**ctrgcn_args,
                   is_use_rank_pool=True).cuda()

inputs = torch.randn(1, 3, 300, 25, 2).cuda()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True,
             with_stack=True) as prof:
    with record_function("model_inference"):
        the_model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
