# validation ratios
nfolds = 5
usage_fold = 1
holdout_fold = .15
unseen_fold = .5

# modeling
nepochs = 200
batch_size = 128
val_batch_size = batch_size * 8
lr = 0.0001
num_heads = 2
num_enc_layers = 2
max_seq = 100
embed_size = 64
dropout = 0.1
dim_ff = 256
