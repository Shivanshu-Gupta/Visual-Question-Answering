import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext import data
from config import parse_args
from model import POSTaggerModel
from train import train_model, test_model

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 300
HIDDEN_DIM = 200


def load_datasets():
    text = data.Field(include_lengths=True)
    tags = data.Field()
    train_data, val_data, test_data = data.TabularDataset.splits(path='RNN_Data_files/', train='train_data.tsv', validation='val_data.tsv', test='val_data.tsv', fields=[('text', text), ('tags', tags)], format='tsv')

    batch_sizes = (args.batch_size, args.batch_size, args.batch_size)
    train_loader, val_loader, test_loader = data.BucketIterator.splits((train_data, val_data, test_data), batch_sizes=batch_sizes, sort_key=lambda x: len(x.text))

    text.build_vocab(train_data)
    tags.build_vocab(train_data)
    dataloaders = {'train': train_loader,
                   'validation': val_loader,
                   'test': val_loader}
    return text, tags, dataloaders


def save_params():
    os.makedirs(args.save_dir, exist_ok=True)
    param_file = args.save_dir + '/' + 'params.pt'
    torch.save(args, param_file)


if __name__ == '__main__':
    global args
    args = parse_args()
    save_params()
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    text, tags, dataloaders = load_datasets()
    text_vocab_size = len(text.vocab.stoi) + 1
    tag_vocab_size = len(tags.vocab.stoi) - 1   # = 42 (not including the <pad> token
    print(text_vocab_size)
    print(tag_vocab_size)

    model = POSTaggerModel(args.rnn_class, EMBEDDING_DIM, HIDDEN_DIM,
                           text_vocab_size, tag_vocab_size, args.use_gpu)
    if args.use_gpu:
            model = model.cuda()

    if args.reload:
        if os.path.isfile(args.reload):
            print("=> loading checkpoint '{}'".format(args.reload))
            checkpoint = torch.load(args.reload)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.reload_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, accuracy {})"
                  .format(args.reload, checkpoint['epoch'], checkpoint['best_acc']))
        else:
            print("=> no checkpoint found at '{}'".format(args.reload))

    if args.test:
        test_model(model, dataloaders['test'], use_gpu=args.use_gpu)
    else:
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        # Decay LR by a factor of gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        print("begin training")
        model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, args.save_dir,
                            num_epochs=args.epochs, use_gpu=args.use_gpu)
