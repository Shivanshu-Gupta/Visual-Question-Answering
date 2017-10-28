import argparse
import yaml

def parse_args():
    print("parsing arguments")
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
    parser.add_argument('--config', metavar='PATH',default='config.yml')
    parser.add_argument('--use_gpu', default=False, action='store_true')

    parser.add_argument('--data_dir', default='RNN_Data_files/', metavar='PATH')
    parser.add_argument('--save_dir', default='/home/cse/dual/cs5130298/scratch/checkpoints2/', metavar='PATH')

    parser.add_argument('--rnn_class', default='lstm',
                        help='class of underlying RNN to use')
    parser.add_argument('--reload', type=str, default='', metavar='PATH',
                        help='path to checkpoint to load (default: none)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='test model on test set (use with --reload)')

    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs to run')

    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--step_size', type=int, default=10, metavar='N')
    parser.add_argument('--gamma', type=float, default=1)

    # parser.add_argument('--wd', default=1e-4, type=float,
    #                     help='weight decay (default: 1e-4)')
    # parser.add_argument('--reg', default=1e-4, type=float,
    #                     help='l2 regularization (default: 1e-4)')
    # parser.add_argument('--optimizer', default='sgd',
    #                     help='optimizer (default: sgd)')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')

    args = parser.parse_args()
    config =yaml.load(open(args.config))
    return (config,args)
    #return args
