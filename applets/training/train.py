# python peripherals
import os
from argparse import ArgumentParser

# numpy
import numpy

# pytorch
import torch
import torch.distributed as dist

# wandb
import wandb

# deep-signature


def fix_random_seeds(seed=30):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)

os.environ["WANDB_DIR"] = os.path.abspath("C:/sweeps")

def main():
    wandb.init(dir=os.getenv("WANDB_DIR", 'C:/sweeps'))
    print('HELLLOOOOOOOOOOOOOOOOOOOO')
    # hyperparameter_defaults = dict(
    #     dropout=0.5,
    #     channels_one=16,
    #     channels_two=32,
    #     batch_size=100,
    #     learning_rate=0.001,
    #     epochs=2,
    # )
    #
    # wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion")

    # config =
    wandb.log({
        'test': 2
    })
    # print(config)


if __name__ == '__main__':
    main()
