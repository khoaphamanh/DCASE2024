import torch

from models import BEATs, BEATsConfig

path = "/home/phamanh/nobackup/DCASE2024/models/pretrained_models/BEATs_iter3.pt"

state_dict = torch.load(path)
# print("model_state_dict[cfg]:", model_state_dict["cfg"])
cfg = BEATsConfig(state_dict["cfg"])
model = BEATs(cfg)
model.load_state_dict(state_dict["model"])
padding_mask = torch.zeros(1, 10000).bool()
# print("padding_mask:", padding_mask)

audio_input_16khz = torch.randn(2, 128, 1201)
labels = model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
print("labels shape:", labels.shape)
print("labels:", labels)
