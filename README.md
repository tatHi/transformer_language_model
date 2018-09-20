# Transformer LM
[UNFINISHED]  
Language model with transformer encoder.  
The source of transformer is forked from:   
https://github.com/jadore801120/attention-is-all-you-need-pytorch  

/data/ : ptb data   
/src/ : source codes  

## Requirement
- python 3.4+
- pytorch 0.4+
see https://pytorch.org/get-started/locally/

- tqdm
```
$ pip install tqdm
```

- numpy
```
$ pip install numpy
```

## Quick Start
Model file is not exported (only evaluation is reported in result folder).
```
$ mkdir result
$ cd src
$ python trainer.py
```

GPU is automatically used when GPU is available.  
To specify GPU ID (as 2):  
```
$ CUDA_VISIBLE_DEVICES=2 python trainer.py
```

Parameters can be given via argparse.
```
$ python trainer.py -d_k 64\
                    -d_v 64\
                    -d_inner 1024\
                    -d_model 512\
                    -n_layers 6\
                    -n_heads 8\
                    -n_warmup_steps 4000\
                    -max_epoch 1000
```

## Model Description
### Encoder
Composed of Encoder Layers.  

hogehoge...
