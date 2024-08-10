# this file contains the hyperparameter of the model

# general hyperparameters
seed = 1998
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
project = "DCASE2024/wav-bu"
data_name_dev = "develop"

# wav2vec no hyperparameter tunning
lr_w2v = 0.0005
emb_w2v = 2048
batch_size_w2v = 20
wd_w2v = 1e-4
epochs_w2v = 100
model_name_w2v = "facebook/wav2vec2-xls-r-300m"

# wav2vec no processor hyperparameter tunning
data_name_np = "develop"
lr_np = 0.00005
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
lr_dev = 0.00008
emb_size_dev = 1024
batch_size_dev = 20
wd_dev = 1e-5
epochs_dev = 50
model_name_dev = "facebook/wav2vec2-xls-r-300m"  #  "facebook/wav2vec2-base-960h"
loss_dev = "adacos"
scale_dev = None
margin_dev = None
optimizer_name_dev = "AdamW"
loss_name_dev = "adacos"
k_dev = 2
distance_dev = "cosine"
percentile_dev = 80
speed_purturb = True
speed_factors = [0.9, 1.1, 1.0, 1.0, 1.0]
classifier_head_dev = False
window_size_dev = None
hop_size_dev = None

# new pipeline develop dataset
project_dev = "DCASE2024/wav-knn"
lr_new = 0.00005
emb_size_new = 1024
batch_size_new = 20
wd_new = 1e-5
epochs_new = 150
model_name_new = "facebook/wav2vec2-xls-r-300m"  # "facebook/wav2vec2-base" "facebook/wav2vec2-base-960h"
loss_new = "adacos"
scale_new = None
margin_new = None
optimizer_name_new = "AdamW"
loss_name_new = "adacos"
k_new = 10
distance_new = "cosine"
percentile_new = 95
speed_purturb_new = True
speed_factors_new = [0.6, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3]
mixup_new = True
mixup_alpha_new = 0.2
classifier_head_new = False
window_size_new = None
hop_size_new = None

# new pipeline mixup
project_dev = "DCASE2024/wav-knn"
lr_new_mixup = 0.00005
emb_size_new_mixup = 1024
batch_size_new_mixup = 100
wd_new_mixup = 1e-5
epochs_new_mixup = 150
model_name_new_mixup = "facebook/wav2vec2-xls-r-300m"  # "facebook/wav2vec2-base" "facebook/wav2vec2-base-960h"
loss_new_mixup = "adacos"
scale_new_mixup = None
margin_new_mixup = None
optimizer_name_new_mixup = "AdamW"
loss_name_new_mixup = "adacos"
k_new_mixup = 10
distance_new_mixup = "cosine"
percentile_new_mixup = 95
speed_purturb_new_mixup = True
speed_factors_new_mixup = [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3]
mixup_new_mixup = True
mixup_alpha_new_mixup = 0.2
classifier_head_new_mixup = False
window_size_new_mixup = None
hop_size_new_mixup = None
dropout_new_mixup = 0.1

# batch uniform sampling
lr_bu = 0.0002
emb_size_bu = 3
batch_size_eval_bu = 100
batch_size_sampler_bu = 20
num_samples_batch_uniform_bu = 14000
wd_bu = 1e-5
epochs_bu = 150
model_name_bu = "facebook/wav2vec2-xls-r-300m"
loss_bu = "adacos"
scale_bu = None
margin_bu = None
optimizer_name_bu = "AdamW"
loss_name_bu = "adacos"
k_bu = 10
distance_bu = "cosine"
threshold_bu = 3
speed_purturb_bu = True
speed_factors_bu = [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3]
mixup_bu = True
mixup_alpha_bu = 0.2
window_size_bu = None
hop_size_bu = None
dropout_bu = 0.0
