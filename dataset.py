# custom Dataset - not used in final implementation.

import os
import torch
from torch.utils.data import Dataset


class POSDataset(Dataset):
    def __init__(self, path, sen_vocab, tag_vocab):
        super(POSDataset, self).__init__()
        self.sen_vocab = sen_vocab
        self.tag_vocab = tag_vocab
        self.num_classes = tag_vocab.size()
        sen_file = os.path.join(path, 'sentences.txt')
        tag_file = os.path.join(path, 'tags.txt')
        self.sentences = []
        with open(sen_file, 'r') as f:
            for line in f:
                idxs = self.sen_vocab.toIdx(line.rstrip('\n').split(' '))
                tensor = torch.LongTensor(idxs)
                self.sentences.append(tensor)

        self.tags = []
        with open(tag_file, 'r') as f:
            for line in f:
                idxs = self.tag_vocab.toIdx(line.rstrip('\n').split(' '))
                tensor = torch.LongTensor(idxs)
                self.tags.append(tensor)

        # making sure there are same number of sentences as tags.
        assert(len(self.sentences) == len(self.tags))

    def __getitem__(self, index):
        sentence = self.sentences[index]
        tags = self.tags[index]
        return sentence, tags

    def __len__(self):
        return len(self.sentences)
