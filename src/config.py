
d_k = 64
d_v = 64
d_word_vec = 512
d_inner = 1024
n_heads = 8
d_model = d_k*n_heads
n_layers = 6
n_warmup_steps = 4000

trainPath = '../data/ptb.train.txt'
validPath = '../data/ptb.valid.txt'
testPath = '../data/ptb.test.txt'
dictPath = '../data/ptb.train.dict'

maxEpoch = 1000

pad_id = 0
