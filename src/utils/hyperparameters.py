# validation ratios
nfolds = 5
usage_fold = 1
holdout_fold = .15
unseen_fold = .5

# modeling
nepochs = 10
batch_size = 512
val_batch_size = batch_size * 2
lr = 0.0001
num_heads = 2
num_enc_layers = 2
max_seq = 100
embed_size = 128
dropout = 0.1
dim_ff = 256
