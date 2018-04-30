from __future__ import print_function, division

import argparse
from solver import Solver
from model import VGG_FCN

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    dest='mode', help='"train" or "eval"',
                    required=True)
parser.add_argument('--vgg-path', type=str,
                    dest='vgg_path', help='path of pretrained vgg16',
                    default='vgg16_weights_tf_dim_ordering_tf_kernels.h5')
parser.add_argument('--model-path', type=str,
                    dest='model_path', help='path of trained FCN model')
parser.add_argument('--train-path', type=str,
                    dest='train_path', help='directory of training data')
parser.add_argument('--val-path', type=str,
                    dest='val_path', help='directory of validation data',
                    default=None)
parser.add_argument('--test-path', type=str,
                    dest='test_path', help='directory of testing data')
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='directory of output')
parser.add_argument('--batch-size', type=int,
                    dest='batch_size', help='batch size',
                    default=4)
parser.add_argument('--learning-rate', type=float,
                    dest='learning_rate', help='learning rate',
                    default=1e-4)
parser.add_argument('--epochs', type=int,
                    dest='epochs', help='number of epochs',
                    default=40)
parser.add_argument('--steps', type=int,
                    dest='steps', help='number of steps',
                    default=400)
args = parser.parse_args()

def main():
    # n_train_data = 12848
    # steps = n_train_data // args.batch_size // 4

    model = VGG_FCN(batch_shape=(512, 512, 3), mode=args.mode,
                    vgg_path=args.vgg_path, model_path=args.model_path)
    solver = Solver(model, batch_size=args.batch_size, epochs=args.epochs,
            steps=args.steps, learning_rate=args.learning_rate,
            train_path=args.train_path, val_path=args.val_path,
            test_path=args.test_path, output_path=args.output_path)
    
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'eval':
        solver.eval()



if __name__ == '__main__':
    main()