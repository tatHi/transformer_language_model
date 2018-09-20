from collections import defaultdict
import pickle
def getDict(data):
    word2id = {'<PAD>':0}
    id2word = {0:'<PAD>'}

    wordCountDict = defaultdict(lambda:0)
    for line in data:
        ws = line.split()
        for w in ws:
            wordCountDict[w] += 1

    wordCountDict['<BOS>'] = len(data)
    wordCountDict['<EOS>'] = len(data)

    for k,v in sorted(wordCountDict.items(), key=lambda x:x[1])[::-1]:
        word2id[k] = len(word2id)
        id2word[word2id[k]] = k

    return word2id, id2word

path = '../data/ptb.train.txt'
dictPath = '../data/ptb.train.dict'

data = [line.strip() for line in open(path) if line.strip()]
pickle.dump(getDict(data), open(dictPath,'wb'))

