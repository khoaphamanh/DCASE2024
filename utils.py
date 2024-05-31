# this file contains the hyperparameter of the model

# general hyperparameters
seed = 1998
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="

# wav2vec no hyperparameter tunning
project = "DCASE2024/wav-test"
lr_w2v = 0.0005
emb_w2v = 2048
batch_size_w2v = 20
wd_w2v = 1e-4
epochs_w2v = 100
model_name_w2v = "facebook/wav2vec2-xls-r-300m"

# wav2vec no processor hyperparameter tunning
project = "DCASE2024/wav-test"
lr_np = 0.00005
emb_np = 1024
batch_size_np = 20
wd_np = 1e-5
epochs_np = 100
model_name_np = "facebook/wav2vec2-xls-r-300m"
loss_np = "arcface"
scale_np = 64
margin_np = 0.5
optimizer_name_np = "AdamW"
loss_name_np = "arcface"
classifier_head_np = False
