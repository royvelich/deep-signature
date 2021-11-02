import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureEuclideanCurvatureTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineCurvatureTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineCurvatureTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureCurvatureNet
from deep_signature.nn.losses import TupletLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils
from argparse import ArgumentParser


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    parser = ArgumentParser()
    parser.add_argument("--group", dest="group")
    parser.add_argument("--epochs", dest="epochs", default=settings.curvature_default_epochs, type=int)
    parser.add_argument("--continue_training", dest="continue_training", default=settings.curvature_default_continue_training, type=bool)
    parser.add_argument("--train_buffer_size", dest="train_buffer_size", default=settings.curvature_default_train_buffer_size, type=int)
    parser.add_argument("--validation_buffer_size", dest="validation_buffer_size", default=settings.curvature_default_validation_buffer_size, type=int)
    parser.add_argument("--train_batch_size", dest="train_batch_size", default=settings.curvature_default_train_batch_size, type=int)
    parser.add_argument("--validation_batch_size", dest="validation_batch_size", default=settings.curvature_default_validation_batch_size, type=int)
    parser.add_argument("--train_dataset_size", dest="train_dataset_size", default=settings.curvature_default_train_dataset_size, type=int)
    parser.add_argument("--validation_dataset_size", dest="validation_dataset_size", default=settings.curvature_default_validation_dataset_size, type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", default=settings.curvature_default_learning_rate, type=float)
    parser.add_argument("--validation_split", dest="validation_split", default=settings.curvature_default_validation_split, type=float)
    parser.add_argument("--sampling_ratio", dest="sampling_ratio", default=settings.curvature_default_sampling_ratio, type=float)
    parser.add_argument("--supporting_points_count", dest="supporting_points_count", default=settings.curvature_default_supporting_points_count, type=int)
    parser.add_argument("--sample_points_count", dest="sample_points_count", default=settings.curvature_default_sample_points_count, type=int)
    parser.add_argument("--multimodality", dest="multimodality", default=settings.curvature_default_multimodality, type=int)
    parser.add_argument("--offset_length", dest="offset_length", default=settings.curvature_default_offset_length, type=int)
    parser.add_argument("--num_workers_train", dest="num_workers_train", default=settings.curvature_default_num_workers_train, type=int)
    parser.add_argument("--num_workers_validation", dest="num_workers_validation", default=settings.curvature_default_num_workers_validation, type=int)
    parser.add_argument("--negative_examples_count", dest="negative_examples_count", default=settings.curvature_default_negative_examples_count, type=int)
    parser.add_argument("--history_size", dest="history_size", default=settings.curvature_default_history_size, type=int)
    args = parser.parse_args()

    OnlineDataset = None
    results_base_dir_path = None
    if args.group == 'euclidean':
        OnlineDataset = DeepSignatureEuclideanCurvatureTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_euclidean_curvature_tuplets_results_dir_path
    elif args.group == 'equiaffine':
        OnlineDataset = DeepSignatureEquiaffineCurvatureTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_equiaffine_curvature_tuplets_results_dir_path
    elif args.group == 'affine':
        OnlineDataset = DeepSignatureAffineCurvatureTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_affine_curvature_tuplets_results_dir_path

    train_dataset = OnlineDataset(
        dataset_size=args.train_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=args.sampling_ratio,
        multimodality=args.multimodality,
        replace=True,
        buffer_size=args.train_buffer_size,
        num_workers=args.num_workers_train,
        supporting_points_count=args.supporting_points_count,
        offset_length=args.offset_length,
        negative_examples_count=args.negative_examples_count)

    validation_dataset = OnlineDataset(
        dataset_size=args.validation_dataset_size,
        dir_path=settings.level_curves_dir_path_validation,
        sampling_ratio=args.sampling_ratio,
        multimodality=args.multimodality,
        replace=False,
        buffer_size=args.validation_buffer_size,
        num_workers=args.num_workers_validation,
        supporting_points_count=args.supporting_points_count,
        offset_length=args.offset_length,
        negative_examples_count=args.negative_examples_count)

    validation_dataset.start()
    validation_dataset.stop()
    train_dataset.start()

    model = DeepSignatureCurvatureNet(sample_points=args.sample_points_count).cuda()
    print(model)

    if args.continue_training:
        latest_subdir = common_utils.get_latest_subdirectory(results_base_dir_path)
        results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
        model.load_state_dict(torch.load(results['model_file_path'], map_location=torch.device('cuda')))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate, line_search_fn='strong_wolfe', history_size=args.history_size)
    curvature_loss_fn = TupletLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[curvature_loss_fn], optimizer=optimizer)
    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_split=args.validation_split,
        results_base_dir_path=results_base_dir_path)
