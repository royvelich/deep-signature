import torch
import numpy
import os
import random
from deep_signature.nn.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureSimilarityArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.trainers import ModelTrainer
from utils import settings
from utils import common as common_utils
from argparse import ArgumentParser
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import builtins
import warnings


def fix_random_seeds(seed=30):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


# def train(gpu, ngpus_per_node, args):
#


if __name__ == '__main__':
    warnings.filterwarnings('error')
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_default_dtype(torch.float64)

    parser = ArgumentParser()
    parser.add_argument("--group")
    parser.add_argument("--epochs", default=settings.arclength_default_epochs, type=int)
    parser.add_argument("--continue-training", default=settings.arclength_default_continue_training, type=bool)
    parser.add_argument("--train-buffer-size", default=settings.arclength_default_train_buffer_size, type=int)
    parser.add_argument("--validation-buffer-size", default=settings.arclength_default_validation_buffer_size, type=int)
    parser.add_argument("--train-batch-size", default=settings.arclength_default_train_batch_size, type=int)
    parser.add_argument("--validation-batch-size", default=settings.arclength_default_validation_batch_size, type=int)
    parser.add_argument("--train-dataset-size", default=settings.arclength_default_train_dataset_size, type=int)
    parser.add_argument("--validation-dataset-size", default=settings.arclength_default_validation_dataset_size, type=int)
    parser.add_argument("--learning-rate", default=settings.arclength_default_learning_rate, type=float)
    parser.add_argument("--validation-split", default=settings.arclength_default_validation_split, type=float)
    parser.add_argument("--supporting-points-count", default=settings.arclength_default_supporting_points_count, type=int)
    parser.add_argument("--anchor-points-count", default=settings.arclength_default_anchor_points_count, type=int)
    parser.add_argument("--multimodality", default=settings.arclength_default_multimodality, type=int)
    parser.add_argument("--min-offset", default=settings.arclength_default_min_offset, type=int)
    parser.add_argument("--max-offset", default=settings.arclength_default_max_offset, type=int)
    parser.add_argument("--num-workers-train", default=settings.arclength_default_num_workers_train, type=int)
    parser.add_argument("--num-workers-validation", default=settings.arclength_default_num_workers_validation, type=int)
    parser.add_argument("--history-size", default=settings.arclength_default_history_size, type=int)
    parser.add_argument("--data-dir", default=settings.data_dir, type=str)

    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--local-rank', default=-1, type=int)

    args = parser.parse_args()

    if "SLURM_PROCID" in os.environ:
        print(f"SLURM_PROCID: {os.environ['SLURM_PROCID']}")

    if "WORLD_SIZE" in os.environ:
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")

    if "MASTER_ADDR" in os.environ:
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    else:
        os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']

    if "MASTER_PORT" in os.environ:
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    # if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
        args.rank = args.local_rank
        args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # dist.init_process_group(backend=args.dist_backend, init_method='tcp://localhost:12000', world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    # if args.rank != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    print(f'args.rank: {args.rank}')
    print(f'args.gpu: {args.gpu}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')

    fix_random_seeds(args.rank)

    OnlineDataset = None
    if args.group == 'euclidean':
        OnlineDataset = DeepSignatureEuclideanArclengthTupletsOnlineDataset
    elif args.group == 'similarity':
        OnlineDataset = DeepSignatureSimilarityArclengthTupletsOnlineDataset
    elif args.group == 'equiaffine':
        OnlineDataset = DeepSignatureEquiaffineArclengthTupletsOnlineDataset
    elif args.group == 'affine':
        OnlineDataset = DeepSignatureAffineArclengthTupletsOnlineDataset

    train_dataset_dir_path = common_utils.get_train_dataset_dir(data_dir=args.data_dir, invariant='arclength', group=args.group)
    validation_dataset_dir_path = common_utils.get_validation_dataset_dir(data_dir=args.data_dir, invariant='arclength', group=args.group)
    results_base_dir_path = common_utils.get_results_dir(data_dir=args.data_dir, invariant='arclength', group=args.group)
    train_curves_dir_path = common_utils.get_train_curves_dir(data_dir=args.data_dir)
    validation_curves_dir_path = common_utils.get_validation_curves_dir(data_dir=args.data_dir)

    train_dataset = OnlineDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        multimodality=args.multimodality,
        replace=True,
        buffer_size=args.train_buffer_size,
        num_workers=args.num_workers_train,
        gpu=0,
        supporting_points_count=args.supporting_points_count,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        anchor_points_count=args.anchor_points_count)

    validation_dataset = OnlineDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=validation_curves_dir_path,
        multimodality=args.multimodality,
        replace=False,
        buffer_size=args.validation_buffer_size,
        num_workers=args.num_workers_validation,
        gpu=0,
        supporting_points_count=args.supporting_points_count,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        anchor_points_count=args.anchor_points_count)

    validation_dataset.load(dataset_dir_path=validation_dataset_dir_path)
    train_dataset.load(dataset_dir_path=train_dataset_dir_path)

    # validation_dataset.start()
    # validation_dataset.stop()
    # train_dataset.start()

    model = DeepSignatureArcLengthNet(sample_points=args.supporting_points_count)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    print('')
    print(model)

    if args.continue_training:
        latest_subdir = common_utils.get_latest_subdirectory(results_base_dir_path)
        results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
        model.load_state_dict(torch.load(results['model_file_path'], map_location=torch.device('cuda')))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate, line_search_fn='strong_wolfe', history_size=args.history_size)
    loss_fn = ArcLengthLoss(anchor_points_count=args.anchor_points_count).cuda(args.gpu)

    model_trainer = ModelTrainer(
        model=model,
        loss_functions=[loss_fn],
        optimizer=optimizer,
        world_size=args.world_size,
        rank=args.rank,
        gpu=args.gpu)

    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_split=args.validation_split,
        results_base_dir_path=results_base_dir_path)


