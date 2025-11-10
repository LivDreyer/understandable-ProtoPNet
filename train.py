import argparse

from ppnet.train_and_test import run_training

# Script arguments
parser = argparse.ArgumentParser(description='Train a new ProtoPNet model', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))

parser.add_argument('--dataset', type=str, required=True, help='path of the dataset to use for training and evaluation')
parser.add_argument('--exp_name', type=str, required=True, help='id of the current experiment')

parser.add_argument('--architecture', type=str, default= 'resnet34', help='model architecture to use as backbone (default: %(default)s)', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'])
parser.add_argument('--num_prototypes', type=int, default=2000, help='number of prototypes to be learned (default: %(default)s)')

parser.add_argument('--epochs', type=int, default=10000, help='number of training epochs (default: %(default)s)')
parser.add_argument('--img_size', type=int, default=224, help='resize dimension for training images (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size (default: %(default)s)')
parser.add_argument('--warm_epochs', type=int, default=150, help='number of warming epochs (default: %(default)s)')
parser.add_argument('--step_size', type=int, default=150, help='step size of the learning rate scheduler (default: %(default)s)')
parser.add_argument('--test_interval', type=int, default=30, help='epoch interval in which to run the model on the test split (default: %(default)s)')
parser.add_argument('--push_interval', type=int, default=300, help='epoch interval in which to push prototypes (default: %(default)s)')

parser.add_argument('--prototype_activation_function', type=str, default='log', choices=['log', 'linear'], help='activation function for the last prototype (default: %(default)s)')
parser.add_argument('--add_on_layers', type=str, default='regular', choices=['regular', 'bottleneck'], help='add-on layers type (default: %(default)s)')
parser.add_argument('--min_diversity', type=float, default=0.1, help='minimum acceptable distance between prototypes (default: %(default)s)')
parser.add_argument('--diversity_coeff', type=float, default=0.1, help='coefficient for intra-class diversity regularization (default: %(default)s)')

parser.add_argument('--gpus', type=str, default='0', help='list of gpus to use, e.g. 0,1,2 (default: %(default)s)')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to use for data loading (default: %(default)s)')
parser.add_argument('--seed', type=int, default=None, help='random seed to use (default: %(default)s)')


if __name__ == '__main__':
    # Start training
    import sys
if len(sys.argv) == 1:  # run col tastino senza argomenti
    sys.argv += [
        '--dataset', 'img/cub200',
        '--exp_name', 'debug_gpu',
        '--architecture', 'resnet34',
        '--epochs', '1',
        '--warm_epochs', '0',
        '--test_interval', '1',
        '--batch_size', '4',
        '--num_workers', '4',
        '--num_prototypes', '600',
        '--gpus', '0'
    ]

    args = parser.parse_args()
    run_training(args)
