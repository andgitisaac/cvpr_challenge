from __future__ import print_function, division

import argparse
from solver import Solver
from model import MultiscaleVGG

parser = argparse.ArgumentParser()

### Model directory ###
parser.add_argument('--vgg-path', type=str,
                    dest='vgg_path', help='path of pretrained vgg16',
                    default='vgg16_weights_tf_dim_ordering_tf_kernels.h5')
parser.add_argument('--model-path', type=str,
                    dest='model_path', help='root dir of loading/saving trained model',
                    default='model/')

### Data directory ###
parser.add_argument('--train-path', type=str,
                    dest='train_path', help='directory of training data')
parser.add_argument('--val-path', type=str,
                    dest='val_path', help='directory of validation data',
                    default=None)
parser.add_argument('--test-path', type=str,
                    dest='test_path', help='directory of testing data')
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='directory of output')

### Mode config ###
parser.add_argument('--mode', type=str,
                    dest='mode', help='"train" or "eval"',
                    required=True)
parser.add_argument('--scale', type=int,
                    dest='scale', help='1 to 5 from coarse to fine',
                    required=True)

### Training config ###
parser.add_argument('--crop-size', type=int,
                    dest='crop_size', help='crop size',
                    default=512)
parser.add_argument('--batch-size', type=int,
                    dest='batch_size', help='batch size',
                    default=3)
parser.add_argument('--learning-rate', type=float,
                    dest='learning_rate', help='learning rate',
                    default=1e-4)
parser.add_argument('--epochs', type=int,
                    dest='epochs', help='number of epochs',
                    default=20)
parser.add_argument('--steps', type=int,
                    dest='steps', help='number of steps',
                    default=400)
args = parser.parse_args()

def main():
    batch_shape = (args.crop_size, args.crop_size, 3)
    # if args.scale == 1:
    #     import_model = args.vgg_path
    # else:
    #     import_model = "{}stream_{}.h5".format(args.model_path, args.scale-1)
    import_model = args.vgg_path
    export_model = "{}_whole".format(args.model_path)
    # .h5 will be added at Solver.train
    # export_model = "{}stream_{}".format(args.model_path, args.scale)


    model = MultiscaleVGG(batch_shape=batch_shape,
                    scale=args.scale,
                    mode=args.mode,
                    import_model=import_model)

    solver = Solver(model,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    export_model=export_model,
                    train_path=args.train_path,
                    val_path=args.val_path,
                    test_path=args.test_path,
                    output_path=args.output_path)
    
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'eval':
        solver.eval()



if __name__ == '__main__':
    main()