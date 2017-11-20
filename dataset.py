import pickle
import torch
import torchvision.transforms as transforms
from IPython.core.debugger import Pdb


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, qafile, img_dir, phase):
        self.examples = pickle.load(open(qafile, 'rb'))
        self.transforms = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        self.image_dir = img_dir
        self.phase = phase

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        _, ques, _, image_id, ans = self.examples[idx]
        # img = Image.open('{0}/{1}2014/COCO_{1}2014_{2:012d}.jpg'.format(self.image_dir, self.phase, image_id))
        # img = img.convert('RGB')
        # img = self.transforms(img)
        emb = torch.load('/home/cse/phd/csz178058/scratch/vqadata/train2014_vqa_i_1024_vgg/{}'.format(image_id))
        img = emb
        return torch.from_numpy(ques).squeeze(), img, ans


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
