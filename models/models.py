import torch

path = "/home/phamanh/nobackup/DCASE2024/models/pretrained_models/BEATs_iter3.pt"

model = torch.load(path)
print("model:", model)
