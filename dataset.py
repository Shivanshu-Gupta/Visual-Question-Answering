import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from IPython.core.debugger import Pdb


class VQADataset(torch.utils.data.Dataset):
    ques_vocab = {}
    ans_vocab = {}

    def __init__(self, data_dir, qafile, img_dir, phase, raw_images=False):
        self.data_dir = data_dir
        self.examples = pickle.load(open(os.path.join(data_dir, qafile), 'rb'))
        if phase == 'train':
            self.load_vocab(data_dir)
        self.transforms = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
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
        _, ques, _, imgid, ans = self.examples[idx]
        if self.raw_images:
            img = Image.open('{0}/{1}/{2}2014/COCO_{2}2014_{3:012d}.jpg'.format(self.data_dir, self.img_dir, self.phase, imgid))
            img = img.convert('RGB')
            img = self.transforms(img)
        else:
            img = torch.load('{}/{}/{}'.format(self.data_dir, self.img_dir, imgid))
        return torch.from_numpy(ques), img, imgid, ans


class VQABatchSampler:
    def __init__(self, data_source, batch_size, drop_last=False):
        self.lengths = [ex[2] for ex in data_source.examples]
        # TODO: Use a better sampling strategy.
        self.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        prev_len = -1
        for idx in self.sampler:
            curr_len = self.lengths[idx]
            if prev_len > 0 and curr_len != prev_len:
                yield batch
                batch = []
            batch.append(idx)
            prev_len = curr_len
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                prev_len = -1
        if len(batch) > 0 and not self.drop_last:
            yield batch
            prev_len = -1

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
