import torch
import numpy
from deep_signature.training.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.training.datasets import DeepSignatureSimilarityArclengthTupletsOnlineDataset
from deep_signature.training.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.training.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from deep_signature.training.networks import ArcLengthNet
from deep_signature.training.losses import ArcLengthLoss
from deep_signature.training.trainers import ModelTrainer
from utils import settings
from deep_signature.core import utils as common_utils
from argparse import ArgumentParser


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)

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

    args = parser.parse_args()

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
    results_base_dir_path = common_utils.get_results_dir(base_dir=args.data_dir, invariant='arclength', group=args.group)
    train_curves_dir_path = common_utils.get_train_curves_dir(data_dir=args.data_dir)
    validation_curves_dir_path = common_utils.get_validation_curves_dir(data_dir=args.data_dir)

    train_dataset = OnlineDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        multimodality=args.multimodality,
        replace=True,
        buffer_size=args.train_buffer_size,
        num_workers=args.num_workers_train,
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
        supporting_points_count=args.supporting_points_count,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        anchor_points_count=args.anchor_points_count)

    validation_dataset.start()
    validation_dataset.stop()
    train_dataset.start()

    model = ArcLengthNet(sample_points=args.supporting_points_count)

    print('')
    print(model)

    if args.continue_training:
        latest_subdir = common_utils.get_latest_subdirectory(results_base_dir_path)
        results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
        model.load_state_dict(torch.load(results['model_file_path'], map_location=torch.device('cuda')))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate, line_search_fn='strong_wolfe', history_size=args.history_size)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ArcLengthLoss(anchor_points_count=args.anchor_points_count)
    model_trainer = ModelTrainer(model=model, loss_function=[loss_fn], optimizer=optimizer)
    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_split=args.validation_split,
        results_base_dir_path=results_base_dir_path)
