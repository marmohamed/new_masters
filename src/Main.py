
from Trainer import *
import argparse

def main(args):
    print('Building the model')
    if args.train_fusion or args.train_end_to_end:
        from model_fusion import Model
        params = {
            'fusion': True
        }
    else :
        from model import Model
        params = {
            'fusion': False
        }
    
    model = Model(graph=None, **params)

    print('Training')

    trainer = ModelTrainer(model=model, data_base_path=args.data_path)


    if args.train_bev:
        print('Train Detection')
        trainer.train_bev(args)


    if args.train_fusion:
        print('Train Fusion')
        trainer.train_fusion(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Object Detection Model')
    parser.add_argument('--batch_size', default=1, help='batch size', type=int)
    parser.add_argument('--epochs', default=100, help='epochs', type=int)
    parser.add_argument('--random_seed', default=0, help='random seed', type=int)
    parser.add_argument('--restore', default=None, help='restore path', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ckpt_path', default=None, type=str)
    

    parser.add_argument('--data_path', default='/Users/apple/Desktop/Master/Data', help='Data path', type=str)
    
    parser.add_argument('--train_bev', default=False, help='train bev', action='store_true')
    parser.add_argument('--train_fusion', default=False, help='train fusion', action='store_true')
    
    args = parser.parse_args()

    main(args)

