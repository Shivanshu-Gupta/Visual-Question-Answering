import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from IPython.core.debugger import Pdb
import numpy as np


class VQADataset(torch.utils.data.Dataset):
    ques_vocab = {}
    ans_vocab = {}

    def __init__(self, data_dir, qafile, img_dir, phase, img_scale=(256, 256), img_crop=224, raw_images=False):
        self.data_dir = data_dir
        self.examples = pickle.load(open(os.path.join(data_dir, qafile), 'rb'))
        #Pdb().set_trace()
        if phase == 'train':
            self.load_vocab(data_dir)
        self.transforms = transforms.Compose([
            transforms.Scale(img_scale),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        self.img_dir = img_dir
        self.phase = phase
        self.raw_images = raw_images    # if true, images and load images, not embeddings

    def load_vocab(self, data_dir):
        ques_vocab_file = os.path.join(data_dir, 'ques_stoi.tsv')
        for line in open(ques_vocab_file):
            parts = line.split('\t')
            tok, idx = parts[0], int(parts[1].strip())
            VQADataset.ques_vocab[idx] = tok
        # NOTE: in version 0.1.1 of torchtext, index 0 is assigned to '<unk>' the first time a unknown token is encountered.
        VQADataset.ques_vocab[0] = '<unk>'
        ans_vocab_file = os.path.join(data_dir, 'ans_itos.tsv')
        for line in open(ans_vocab_file):
            parts = line.split('\t')
            VQADataset.ans_vocab[parts[0]] = parts[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ques_id, ques, _, imgid, ans = self.examples[idx]
        if self.raw_images:
            img = Image.open('{0}/{1}/COCO_{1}_{2:012d}.jpg'.format(self.data_dir, self.img_dir, imgid))
            img = img.convert('RGB')
            img = self.transforms(img)
        else:
            img = torch.load('{}/{}/{}'.format(self.data_dir, self.img_dir, imgid))
        return torch.from_numpy(ques), img, imgid, ans, ques_id


class RandomSampler:
    def __init__(self,data_source,batch_size):
        self.lengths = [ex[2] for ex in data_source.examples]
        self.batch_size = batch_size

    def randomize(self):
        #random.shuffle(
        N = len(self.lengths)
        self.ind = np.arange(0,len(self.lengths))
        np.random.shuffle(self.ind)
        self.ind = list(self.ind)
        self.ind.sort(key = lambda x: self.lengths[x])
        self.block_ids = {}
        random_block_ids = list(range(N))
        np.random.shuffle(random_block_ids)
        #generate a random number between 0 to N - 1
        blockid = random_block_ids[0]
        self.block_ids[self.ind[0]] = blockid
        running_count = 1 
        for ind_it in range(1,N):
            if running_count >= self.batch_size or self.lengths[self.ind[ind_it]] != self.lengths[self.ind[ind_it-1]]:
                blockid = random_block_ids[ind_it]
                running_count = 0 
            #   
            self.block_ids[self.ind[ind_it]] = blockid
            running_count += 1
        #  
        # Pdb().set_trace()
        self.ind.sort(key = lambda x: self.block_ids[x])
         

    def __iter__(self):
        # Pdb().set_trace()
        self.randomize()
        return iter(self.ind)

    def __len__(self):
        return len(self.ind)

class VQABatchSampler:
    def __init__(self, data_source, batch_size, drop_last=False):
        self.lengths = [ex[2] for ex in data_source.examples]
        # TODO: Use a better sampling strategy.
        # self.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
        self.sampler = RandomSampler(data_source,batch_size)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.unk_emb = 1000

    def __iter__(self):
        batch = []
        prev_len = -1
        this_batch_counter = 0
        for idx in  self.sampler:
            if self.data_source.examples[idx][4] == self.unk_emb:
                continue
            #
            curr_len = self.lengths[idx]
            if prev_len > 0 and curr_len != prev_len:
                yield batch
                batch = []
                this_batch_counter = 0
            #
            batch.append(idx)
            prev_len = curr_len
            this_batch_counter += 1
            if this_batch_counter == self.batch_size:
                yield batch
                batch = []
                prev_len = -1
                this_batch_counter = 0
        #
        if len(batch) > 0 and not self.drop_last:
            yield batch
            #self.sampler.randomize()
            prev_len = -1
            this_batch_counter = 0

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
