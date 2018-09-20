import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import Models as M
import numpy as np

import config


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(config.pad_id)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(config.pad_id)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=config.pad_id, reduction='sum')
    return loss

class LM(nn.Module):
    def __init__(self, vocSize, maxSeqLen, padId):
        super().__init__()
       
        '''
        d_k = 32
        d_v = 32
        d_word_vec = 256
        d_model = 256
        d_inner = 512
        n_heads = 8
        n_layers = 6
        '''

        #self.embed = nn.Embedding(vocSize, d_model, padding_idx=padId)
        self.encoder = M.Encoder(n_src_vocab=vocSize,
                                 len_max_seq=maxSeqLen,
                                 d_word_vec=config.d_word_vec,
                                 n_layers=config.n_layers,
                                 n_heads=config.n_heads,
                                 d_k=config.d_k,
                                 d_v=config.d_v,
                                 d_model=config.d_model,
                                 d_inner=config.d_inner,
                                 dropout=0.1)
        self.outputDropout = nn.Dropout(0.5)
        self.outputLinear = nn.Linear(config.d_model, vocSize, bias=False)
        nn.init.xavier_normal_(self.outputLinear.weight)

        self.padId = padId
        self.use_cuda = False
        self.tensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

        #self.criterion = nn.CrossEntropyLoss(ignore_index=0,reduction='sum')
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, idLines, inputLens, return_attn=False):
        # masking
        idLines, inputLens = extendSequence(idLines, inputLens, self.padId)

        idLines_var = Variable(self.tensor(idLines))
        inputLens_tensor = self.tensor(inputLens)

        enc_outputs = self.encoder.forward(idLines_var, inputLens_tensor, return_attn)

        # extracting
        # inputLensに応じて最後のベクトルだけ取ってくる
        #ys = enc_outputs[[i for i in range(len(inputLens))], [max(il)-1 for il in inputLens]]
        
        # tanh
        ys = enc_outputs
        
        ys = self.outputDropout(ys)

        zs = self.outputLinear(ys)
        zs = torch.matmul(ys, self.encoder.src_word_emb.weight.transpose(1,0))

        return zs

    def getLoss(self, idLines):
        xs,ts,inputLens = makeXTlens(idLines,self.padId)
        ts = Variable(self.tensor(ts))
        zs = self.forward(xs,inputLens)

        zs = zs.view(-1,zs.size(2))
        ts = ts.contiguous().view(-1)

        #loss = F.cross_entropy(zs,ts,ignore_index=0)
        #loss = self.criterion(zs,ts)
        
        loss, n_correct = cal_performance(zs, ts, smoothing=False)
        
        return loss, n_correct, ts

    def getProbs(self, idLines):
        xs,ts,inputLens = makeXTlens(idLines,self.padId)
        zs = self.forward(xs,inputLens)
        zs = zs.view(zs.shape[0]*zs.shape[1],-1)

        probs = F.softmax(zs, dim=1)
        log_probs = torch.log(probs) 

        # paddingを弾く
        ts = [j for t in ts for j in t]
        x_indices = [i for i in range(log_probs.shape[0]) if ts[i]!=0]
        t_indices = [t for t in ts if t!=0]
        
        ps = log_probs[x_indices, t_indices]
        
        preds = torch.argmax(log_probs[x_indices,],dim=-1).data

        size = ps.shape[0]
        ps = -torch.sum(ps)

        return ps, size, preds

def makeXTlens(idLines, padId):
    xs = [idLine[:-1] for idLine in idLines]
    inputLens = [len(x) for x in xs]
    ml = max(inputLens)
    ts = [idLine[1:]+[padId]*(ml-len(idLine)+1) for idLine in idLines]

    return xs, ts, inputLens

def extendSequence(idLines, inputLens, padId):
    '''
    input:
        idLines = [[1,2],[1,2,3]]
        inputLens = [2,3]
        padId = 0
    output:
        neoIdLines
        [[1, 2, 0]
         [1, 2, 3]]

        neoLength
        [[1,2,0]
         [1,2,3]]
    '''
    maxLength = max(inputLens)

    neoLines = []
    neoLength = []
    for idLine, inputLen in zip(idLines, inputLens):
        neoLines.append(idLine + [padId]*(maxLength-inputLen))
        neoLength.append(list(range(1,inputLen+1)) + [padId]*(maxLength-inputLen))

    return neoLines, neoLength

if __name__ == '__main__':
    import numpy as np
    import torch.optim as optim

    lm = LM(3, 120, 0)
    opt = optim.Adam(lm.parameters())

    idLines = [[1,2,1,2,1], [1,2,1]]
    p = lm.getProbs(idLines)
    for i in range(20):    
        input()
        opt.zero_grad()
        
        loss = lm.getLoss(idLines)
        print(loss)
        
        loss.backward()
        opt.step()

        p = lm.getProbs(idLines)
