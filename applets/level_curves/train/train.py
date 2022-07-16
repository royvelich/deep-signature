# python peripherals
import os
import random
from argparse import ArgumentParser
import builtins
import warnings

# numpy
import numpy

# pytorch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

# deep-signature
from deep_signature.nn.datasets import CurvatureTupletsDataset
from deep_signature.nn.datasets import ArclengthTupletsDataset
from deep_signature.nn.datasets import DifferentialInvariantsTupletsDataset
from deep_signature.nn.networks import CurvatureNet
from deep_signature.nn.networks import ArcLengthNet
from deep_signature.nn.networks import DifferentialInvariantsNet
from deep_signature.nn.networks import DifferentialInvariantsNetBYOL
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.losses import DifferentialInvariantsLoss
from deep_signature.nn.trainers import ModelTrainer
from utils import settings
from utils import common as common_utils


def fix_random_seeds(seed=30):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def create_curvature_environment(args):
    train_dataset = CurvatureTupletsDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        replace=True,
        buffer_size=args.train_dataset_size,
        num_workers=args.num_workers_train,
        args=args)

    validation_dataset = CurvatureTupletsDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=validation_curves_dir_path,
        replace=False,
        buffer_size=args.validation_dataset_size,
        num_workers=args.num_workers_validation,
        args=args)

    model = CurvatureNet(sample_points=common_utils.get_sample_points_count(args=args))
    loss_fn = CurvatureLoss()
    return train_dataset, validation_dataset, model, loss_fn


def create_arclength_environment(args):
    train_dataset = ArclengthTupletsDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        replace=True,
        buffer_size=args.train_dataset_size,
        num_workers=args.num_workers_train,
        args=args)

    validation_dataset = ArclengthTupletsDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=validation_curves_dir_path,
        replace=False,
        buffer_size=args.validation_dataset_size,
        num_workers=args.num_workers_validation,
        args=args)

    model = ArcLengthNet(sample_points=common_utils.get_sample_points_count(args=args))
    loss_fn = ArcLengthLoss(anchor_points_count=args.anchor_points_count)
    return train_dataset, validation_dataset, model, loss_fn


def create_differential_invariants_environment(args):
    train_dataset = DifferentialInvariantsTupletsDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        replace=True,
        buffer_size=args.train_dataset_size,
        num_workers=args.num_workers_train,
        args=args)

    validation_dataset = DifferentialInvariantsTupletsDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=validation_curves_dir_path,
        replace=False,
        buffer_size=args.validation_dataset_size,
        num_workers=args.num_workers_validation,
        args=args)

    model = DifferentialInvariantsNetBYOL(sample_points=common_utils.get_sample_points_count(args=args))
    loss_fn = DifferentialInvariantsLoss()
    return train_dataset, validation_dataset, model, loss_fn


def create_environment(args):
    if args.invariant == 'curvature':
        return create_curvature_environment(args=args)
    elif args.group == 'arclength':
        return create_arclength_environment(args=args)

    return create_differential_invariants_environment(args=args)


if __name__ == '__main__':
    # torch.set_default_dtype(torch.float64)

    parser = ArgumentParser()

    # data settings
    parser.add_argument("--group", type=str)
    parser.add_argument("--invariant", type=str)
    parser.add_argument("--data-dir", default=settings.data_dir, type=str)

    # training settings
    parser.add_argument("--validation-split", default=settings.default_validation_split, type=float)
    parser.add_argument("--train-batch-size", default=settings.default_train_batch_size, type=int)
    parser.add_argument("--validation-batch-size", default=settings.default_validation_batch_size, type=int)
    parser.add_argument("--learning-rate", default=settings.default_learning_rate, type=float)
    parser.add_argument("--epochs", default=settings.default_epochs, type=int)
    parser.add_argument("--history-size", default=settings.default_history_size, type=int)

    # dataset size
    parser.add_argument("--train-dataset-size", default=settings.default_train_dataset_size, type=int)
    parser.add_argument("--validation-dataset-size", default=settings.default_validation_dataset_size, type=int)

    # buffer size (online dataset)
    parser.add_argument("--train-buffer-size", default=settings.default_train_buffer_size, type=int)
    parser.add_argument("--validation-buffer-size", default=settings.default_validation_buffer_size, type=int)

    # number of worker threads (online dataset)
    parser.add_argument("--num-workers-train", default=settings.default_num_workers_train, type=int)
    parser.add_argument("--num-workers-validation", default=settings.default_num_workers_validation, type=int)

    # curvature settings
    parser.add_argument("--supporting-points-count", default=settings.default_supporting_points_count, type=int)
    parser.add_argument("--sampling-ratio", default=settings.default_sampling_ratio, type=float)
    parser.add_argument("--offset-length", default=settings.default_offset_length, type=int)
    parser.add_argument("--multimodality", default=settings.default_multimodality, type=int)
    parser.add_argument("--negative-examples-count", default=settings.default_negative_examples_count, type=int)

    # arc-length settings
    parser.add_argument("--anchor-points-count", default=settings.default_anchor_points_count, type=int)
    parser.add_argument("--min-offset", default=settings.default_min_offset, type=int)
    parser.add_argument("--max-offset", default=settings.default_max_offset, type=int)

    # parallel training
    # parser.add_argument('--gpu', default=None, type=int)
    # parser.add_argument('--world-size', default=-1, type=int)
    # parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='gloo')
    # parser.add_argument('--local-rank', default=-1, type=int)

    args = parser.parse_args()

    train_dataset_dir_path = common_utils.get_train_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    validation_dataset_dir_path = common_utils.get_validation_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    results_base_dir_path = common_utils.get_results_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    train_curves_dir_path = common_utils.get_train_curves_dir(data_dir=args.data_dir)
    validation_curves_dir_path = common_utils.get_validation_curves_dir(data_dir=args.data_dir)

    # # if args.distributed:
    # if args.local_rank != -1: # for torch.distributed.launch
    #     args.rank = args.local_rank
    #     args.gpu = args.local_rank
    # elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()

    # dist.barrier()

    slurm_props = ['SLURM_GPUS_PER_NODE',
                   'SLURM_GPUS_ON_NODE',
                   'SLURM_GPUS',
                   'SLURM_JOB_CPUS_PER_NODE',
                   'SLURM_JOB_NAME',
                   'SLURM_JOBID',
                   'SLURM_NODEID',
                   'SLURM_PROCID',
                   'SLURM_CPUS_PER_TASK',
                   'SLURM_TASK_PID',
                   'SLURM_NODELIST',
                   'SLURM_MEM_PER_NODE',
                   'SLURM_JOB_PARTITION',
                   'SLURM_JOB_NUM_NODES',
                   'SLURM_JOB_ACCOUNT',
                   'SLURM_GPUS_PER_TASK',
                   'SLURM_MEM_PER_CPU',
                   'SLURM_MEM_PER_GPU',
                   'SLURM_NODE_ALIASES',
                   'SLURM_NTASKS',
                   'SLURM_NTASKS_PER_CORE',
                   'SLURM_NTASKS_PER_GPU',
                   'SLURM_NTASKS_PER_NODE',
                   'SLURM_STEP_GPUS',
                   'SLURM_LAUNCH_NODE_IPADDR',
                   'SLURM_SUBMIT_HOST',
                   'MASTER_ADDR',
                   'MASTER_PORT']

    print('---------------------------------------------------------------')
    if 'SLURM_PROCID' in os.environ:
        print(f"SLURM PROPERTIES, SLURM_PROCID: {os.environ['SLURM_PROCID']}:")
    else:
        print("SLURM PROPERTIES")
    print('---------------------------------------------------------------')
    for slurm_prop in slurm_props:
        if slurm_prop in os.environ:
            print(f"os.environ['{slurm_prop}']: {os.environ[slurm_prop]}")
        else:
            print(f"os.environ['{slurm_prop}'] not defined!")

    # if 'SLURM_LAUNCH_NODE_IPADDR' in os.environ:
    #     os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    #     print(f"os.environ['MASTER_ADDR']: {os.environ['MASTER_ADDR']}")

    # print('---------------------------------------------------------------')
    # print(f"DIST PROPERTIES, SLURM_PROCID: {os.environ['SLURM_PROCID']}:")
    # print('---------------------------------------------------------------')
    # print(f'args.rank: {args.rank}')
    # print(f'args.gpu: {args.gpu}')
    # print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    # print(f'dist.get_backend: {dist.get_backend()}')
    # print(f'dist.get_rank: {dist.get_rank()}')
    # print(f'dist.world_size: {dist.get_world_size()}')
    # print('---------------------------------------------------------------')

    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    else:
        rank = 0

    if 'SLURM_LOCALID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        local_rank = 0

    if 'SLURM_NTASKS' in os.environ:
        size = int(os.environ['SLURM_NTASKS'])
    else:
        size = 1

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        cpus_per_task = 1

    if 'SLURM_STEP_GPUS' in os.environ:
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    else:
        gpu_ids = [0]

    if 'SLURM_SUBMIT_HOST' in os.environ:
        os.environ['MASTER_ADDR'] = os.environ['SLURM_SUBMIT_HOST']
        os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids)))

    print(f'rank: {rank}')
    print(f'local_rank: {local_rank}')
    print(f'size: {size}')
    print(f'cpus_per_task: {cpus_per_task}')
    print(f'gpu_ids: {gpu_ids}')
    print(f"os.environ['MASTER_ADDR']: {os.environ['MASTER_ADDR']}")
    print(f"os.environ['MASTER_PORT']: {os.environ['MASTER_PORT']}")

    dist.init_process_group(backend=args.dist_backend, init_method='env://', world_size=size, rank=rank)

    torch.cuda.set_device(local_rank)


    train_dataset, validation_dataset, model, loss_fn = create_environment(args=args)
    # validation_dataset.load(dataset_dir_path=validation_dataset_dir_path)
    # train_dataset.load(dataset_dir_path=train_dataset_dir_path)

    # validation_dataset.load(dataset_dir_path=validation_dataset_dir_path)
    # train_dataset.load(dataset_dir_path=train_dataset_dir_path)

    validation_dataset.start()
    validation_dataset.stop()
    train_dataset.start()
    # train_dataset.stop()

    # validation_dataset.save(dataset_dir_path=validation_dataset_dir_path)
    # train_dataset.save(dataset_dir_path=train_dataset_dir_path)

    device = torch.device("cuda")
    model.cuda(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # if rank == 0:
    print(f'RANK: {rank}')
    print('')
    print(model)

    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate, line_search_fn='strong_wolfe', history_size=args.history_size)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model_trainer = ModelTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        world_size=size,
        rank=rank,
        device=device)

    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_split=args.validation_split,
        results_base_dir_path=results_base_dir_path)