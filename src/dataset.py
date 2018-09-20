from collections import defaultdict
import pickle

class Dataset:
    def __init__(self, textPath, dictPath=None):
        # path: data path
        # raw_data
        data = [line.strip() for line in open(textPath) if line.strip()]
        self.data = data

        if dictPath is None:
            # dicts
            # id 0 is for padding
            self.id2word = {0:'<PAD>'}
            self.word2id = {'<PAD>':0}

            # set dict
            self.setDict(data)
        else:
            self.word2id, self.id2word = pickle.load(open(dictPath,'rb'))
        
        self.idData = []

        self.setIdData(data)

    def setDict(self, data):
        wordCountDict = defaultdict(lambda:0)
        for line in data:
            ws = line.split()
            for w in ws:
                wordCountDict[w] += 1

        wordCountDict['<BOS>'] = len(data)
        wordCountDict['<EOS>'] = len(data)

        for k,v in sorted(wordCountDict.items(), key=lambda x:x[1])[::-1]:
            self.word2id[k] = len(self.word2id)
            self.id2word[self.word2id[k]] = k

    def setIdData(self, data):
        bos = '<BOS>'
        eos = '<EOS>'
        for line in data:
            ws = [bos] + line.split() + [eos]
            self.idData.append(self.words2ids(ws))

    def words2ids(self, words):
        ids = [self.word2id[word] for word in words]
        return ids

    def ids2words(self, ids):
        words = [self.id2word[i] for i in ids]
        return words

if __name__ == '__main__':
    path = '../data/ptb.train.txt'
    ds = Dataset(path)
    for line in ds.idData:
        print(line)
