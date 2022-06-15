import torch

tmp=torch.load('./cks/fed_medmnist_metafed_0.1_0.1_0.001_0.5_0_0.1_0.0_5_1/metafed')
print(tmp.keys())
print(tmp['server_model'])