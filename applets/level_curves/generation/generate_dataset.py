from deep_signature.nn.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureSimilarityArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEuclideanCurveTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureSimilarityCurveTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineCurveTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineCurveTupletsOnlineDataset
from utils import settings
from utils import common as common_utils
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--group", type=str)
    parser.add_argument("--invariant", type=str)
    parser.add_argument("--train-dataset-size", default=settings.arclength_default_train_dataset_size, type=int)
    parser.add_argument("--validation-dataset-size", default=settings.arclength_default_validation_dataset_size, type=int)
    parser.add_argument("--supporting-points-count", default=settings.arclength_default_supporting_points_count, type=int)
    parser.add_argument("--anchor-points-count", default=settings.arclength_default_anchor_points_count, type=int)
    parser.add_argument("--multimodality", default=settings.arclength_default_multimodality, type=int)
    # parser.add_argument("--min-offset", default=settings.arclength_default_min_offset, type=int)
    # parser.add_argument("--max-offset", default=settings.arclength_default_max_offset, type=int)
    parser.add_argument("--num-workers-train", default=settings.arclength_default_num_workers_train, type=int)
    parser.add_argument("--num-workers-validation", default=settings.arclength_default_num_workers_validation, type=int)
    parser.add_argument("--sampling-ratio", type=float)
    parser.add_argument("--offset-length", type=int)
    parser.add_argument("--negative-examples-count", type=int)
    parser.add_argument("--data-dir", default=settings.data_dir, type=str)

    online_datasets = {}
    online_datasets['euclidean'] = {}
    online_datasets['similarity'] = {}
    online_datasets['equiaffine'] = {}
    online_datasets['affine'] = {}
    online_datasets['euclidean']['arclength'] = DeepSignatureEuclideanArclengthTupletsOnlineDataset
    online_datasets['euclidean']['curve'] = DeepSignatureEuclideanCurveTupletsOnlineDataset
    online_datasets['similarity']['arclength'] = DeepSignatureSimilarityArclengthTupletsOnlineDataset
    online_datasets['similarity']['curve'] = DeepSignatureSimilarityCurveTupletsOnlineDataset
    online_datasets['equiaffine']['arclength'] = DeepSignatureEquiaffineArclengthTupletsOnlineDataset
    online_datasets['equiaffine']['curve'] = DeepSignatureEquiaffineCurveTupletsOnlineDataset
    online_datasets['affine']['arclength'] = DeepSignatureAffineArclengthTupletsOnlineDataset
    online_datasets['affine']['curve'] = DeepSignatureAffineCurveTupletsOnlineDataset

    args = parser.parse_args()
    OnlineDataset = online_datasets[args.group][args.invariant]

    train_dataset_dir_path = common_utils.get_train_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    validation_dataset_dir_path = common_utils.get_validation_dataset_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    results_base_dir_path = common_utils.get_results_dir(data_dir=args.data_dir, invariant=args.invariant, group=args.group)
    train_curves_dir_path = common_utils.get_train_curves_dir(data_dir=args.data_dir)
    validation_curves_dir_path = common_utils.get_validation_curves_dir(data_dir=args.data_dir)
    min_offset = common_utils.get_arclength_default_min_offset(supporting_points_count=args.supporting_points_count)
    max_offset = common_utils.get_arclength_default_max_offset(supporting_points_count=args.supporting_points_count)

    if args.invariant == 'arclength':
        train_dataset = OnlineDataset(
            dataset_size=args.train_dataset_size,
            dir_path=train_curves_dir_path,
            multimodality=args.multimodality,
            replace=False,
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
    elif args.invariant == 'curve':
        train_dataset = OnlineDataset(
            dataset_size=args.train_dataset_size,
            dir_path=train_curves_dir_path,
            sampling_ratio=args.sampling_ratio,
            multimodality=args.multimodality,
            replace=False,
            buffer_size=args.train_dataset_size,
            num_workers=args.num_workers_train,
            supporting_points_count=args.supporting_points_count,
            offset_length=args.offset_length,
            negative_examples_count=args.negative_examples_count)

        validation_dataset = OnlineDataset(
            dataset_size=args.validation_dataset_size,
            dir_path=validation_curves_dir_path,
            sampling_ratio=args.sampling_ratio,
            multimodality=args.multimodality,
            replace=False,
            buffer_size=args.validation_dataset_size,
            num_workers=args.num_workers_validation,
            supporting_points_count=args.supporting_points_count,
            offset_length=args.offset_length,
            negative_examples_count=args.negative_examples_count)

    train_dataset.start()
    train_dataset.save(dataset_dir_path=train_dataset_dir_path)
    train_dataset.stop()

    validation_dataset.start()
    validation_dataset.save(dataset_dir_path=validation_dataset_dir_path)
    validation_dataset.stop()
