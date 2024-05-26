# this file contains the hyperparameter of the model

# general hyperparameters
seed = 1998
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="

# wav2vec no hyperparameter tunning
project = "DCASE2024/wav-test"
lr_w2v = 5e-4
emb_w2v = 2048
batch_size_w2v = 32
wd_w2v = 1e-4
epochs_w2v = 100
model_name_w2v = "facebook/wav2vec2-xls-r-300m"

min_rate_w2v = 0.8
