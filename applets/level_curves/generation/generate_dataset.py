from deep_signature.nn.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureSimilarityArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from utils import settings
from utils import common as common_utils
from argparse import ArgumentParser

if __name__ == '__main__':
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
    # parser.add_argument("--validation-split", default=settings.arclength_default_validation_split, type=float)
    parser.add_argument("--supporting-points-count", default=settings.arclength_default_supporting_points_count, type=int)
    parser.add_argument("--anchor-points-count", default=settings.arclength_default_anchor_points_count, type=int)
    parser.add_argument("--multimodality", default=settings.arclength_default_multimodality, type=int)
    # parser.add_argument("--min-offset", default=settings.arclength_default_min_offset, type=int)
    # parser.add_argument("--max-offset", default=settings.arclength_default_max_offset, type=int)
    parser.add_argument("--num-workers-train", default=settings.arclength_default_num_workers_train, type=int)
    parser.add_argument("--num-workers-validation", default=settings.arclength_default_num_workers_validation, type=int)
    parser.add_argument("--history-size", default=settings.arclength_default_history_size, type=int)
    parser.add_argument("--data-dir", default=settings.data_dir, type=str)
    parser.add_argument("--invariant", type=str)

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

    train_dataset_dir_path = common_utils.get_train_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    validation_dataset_dir_path = common_utils.get_validation_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    results_base_dir_path = common_utils.get_results_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    train_curves_dir_path = common_utils.get_train_curves_dir(data_dir=args.data_dir)
    validation_curves_dir_path = common_utils.get_validation_curves_dir(data_dir=args.data_dir)
    min_offset = common_utils.get_arclength_default_min_offset(supporting_points_count=args.supporting_points_count)
    max_offset = common_utils.get_arclength_default_max_offset(supporting_points_count=args.supporting_points_count)

    train_dataset = OnlineDataset(
        dataset_size=args.train_dataset_size,
        dir_path=train_curves_dir_path,
        multimodality=args.multimodality,
        replace=True,
        buffer_size=args.train_buffer_size,
        num_workers=args.num_workers_train,
        supporting_points_count=args.supporting_points_count,
        min_offset=min_offset,
        max_offset=max_offset,
        anchor_points_count=args.anchor_points_count)

    validation_dataset = OnlineDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=validation_curves_dir_path,
        multimodality=args.multimodality,
        replace=False,
        buffer_size=args.validation_buffer_size,
        num_workers=args.num_workers_validation,
        supporting_points_count=args.supporting_points_count,
        min_offset=min_offset,
        max_offset=max_offset,
        anchor_points_count=args.anchor_points_count)

    train_dataset.start()
    train_dataset.save(dataset_dir_path=train_dataset_dir_path)
    train_dataset.stop()

    validation_dataset.start()
    validation_dataset.save(dataset_dir_path=validation_dataset_dir_path)
    validation_dataset.stop()
