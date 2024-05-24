from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from data.preprocessing import DataPreprocessing, raw_data_path
import numpy as np
from torchinfo import summary
import torch

data_preprocessing = DataPreprocessing(raw_data_path=raw_data_path)

train_data, train_label = data_preprocessing.load_data(train=True, test=False)
fs = data_preprocessing.fs

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Load the model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-300m")

test = train_data[0:5]
print("test shape:", test.shape)
# summary(model)

test = processor(test, return_tensors="pt", sampling_rate=fs).input_values
print("test shape:", test.shape)
print("test:", test)
print("test:", test.dtype)
test = test.to(torch.float32)

with torch.inference_mode():
    out = model(test)
    print("out shape:", out.logits.shape)
