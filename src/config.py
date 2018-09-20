import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-d_k',type=int,default=64)
parser.add_argument('-d_v',type=int,default=64)
parser.add_argument('-d_inner',type=int,default=1024)
parser.add_argument('-d_model',type=int,default=512)
parser.add_argument('-n_layers',type=int,default=6)
parser.add_argument('-n_heads',type=int,default=8)
parser.add_argument('-n_warmup_steps',type=int,default=4000)
parser.add_argument('-max_epoch',type=int,default=1000)

args = parser.parse_args()

d_k = args.d_k
d_v = args.d_v
d_inner = args.d_inner
n_heads = args.n_heads
d_model = args.d_model # recomended (d_k * n_heads)
d_word_vec = args.d_model
n_layers = args.n_layers
n_warmup_steps = args.n_warmup_steps
maxEpoch = args.max_epoch

trainPath = '../data/ptb.train.txt'
validPath = '../data/ptb.valid.txt'
testPath = '../data/ptb.test.txt'
dictPath = '../data/ptb.train.dict'

pad_id = 0
