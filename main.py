import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import VQADataset, VQABatchSampler
from train import train_model
import vqa
import san
from IPython.core.debugger import Pdb
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 300
HIDDEN_DIM = 200


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.yml')


def load_datasets(data_dir, phases, img_emb_dir):
    # Pdb().set_trace()
    datasets = {x: VQADataset('{}/{}_data.pkl'.format(data_dir, x), img_emb_dir, x) for x in phases}
    batch_samplers = {x: VQABatchSampler(datasets[x], 32) for x in phases}
    dataloaders = {x: DataLoader(datasets[x], batch_sampler=batch_samplers[x], num_workers=4) for x in phases}
    dataset_sizes = {x: len(datasets[x]) for x in phases}
    print(dataset_sizes)
    return dataloaders


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    args.config = os.path.join(os.getcwd(), args.config)
    config = yaml.load(open(args.config))
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    phases = ['train', 'val']
    dataloaders = load_datasets('datasets', phases, img_emb_dir='/scratch/cse/phd/csz178058/vqadata/')

    config['model']['params']['vocab_size'] = 22226 + 1     # +1 to include '<unk>'
    config['model']['params']['output_size'] = 1001

    if config['model_class'] == 'vqa':
        model = vqa.VQAModel(**config['model']['params'])
    elif config['model_class'] == 'san':
        model = san.SANModel(**config['model']['params'])
    print(model)
    criterion = nn.CrossEntropyLoss()

    if config['optim']['class'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              **config['optim']['params'])
    elif config['optim']['class'] == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                  **config['optim']['params'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               **config['optim']['params'])

    if config['use_gpu']:
            model = model.cuda()
    # Decay LR by a factor of gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("begin training")
    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, '/scratch/cse/dual/cs5130298/vqa',
                        num_epochs=25, use_gpu=config['use_gpu'])
