# this file contains the hyperparameter of the model

# general hyperparameters
seed = 1998
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
project = "DCASE2024/wav-test"
data_name_dev = "develop"

# wav2vec no hyperparameter tunning
project = "DCASE2024/wav-test"
lr_w2v = 0.0005
emb_w2v = 2048
batch_size_w2v = 20
wd_w2v = 1e-4
epochs_w2v = 100
model_name_w2v = "facebook/wav2vec2-xls-r-300m"

# wav2vec no processor hyperparameter tunning
data_name_np = "develop"
lr_np = 0.00008
emb_size_np = 1024
batch_size_np = 80
wd_np = 1e-5
epochs_np = 100
model_name_np = "facebook/wav2vec2-xls-r-300m"
loss_np = "adacos"
scale_np = 30
margin_np = 0.5
optimizer_name_np = "AdamW"
loss_name_np = "adacos"
classifier_head_np = False
window_size_np = 8000
hop_size_np = 8000


# wav2vec and knn for develop dataset
project_dev = "DCASE2024/wav-knn"
lr_dev = 0.00005
emb_size_dev = 1024
batch_size_dev = 20
wd_dev = 1e-5
epochs_dev = 50
model_name_dev = "facebook/wav2vec2-xls-r-300m"
loss_dev = "adacos"
scale_dev = None
margin_dev = None
optimizer_name_dev = "AdamW"
loss_name_dev = "adacos"
k_dev = 2
distance_dev = "cosine"
percentile_dev = 95
classifier_head_dev = False
window_size_dev = None
hop_size_dev = None
