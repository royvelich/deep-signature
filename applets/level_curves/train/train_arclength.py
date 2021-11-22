import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils
from argparse import ArgumentParser


if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    parser = ArgumentParser()
    parser.add_argument("--group", dest="group")
    parser.add_argument("--epochs", dest="epochs", default=settings.arclength_default_epochs, type=int)
    parser.add_argument("--continue_training", dest="continue_training", default=settings.arclength_default_continue_training, type=bool)
    parser.add_argument("--train_buffer_size", dest="train_buffer_size", default=settings.arclength_default_train_buffer_size, type=int)
    parser.add_argument("--validation_buffer_size", dest="validation_buffer_size", default=settings.arclength_default_validation_buffer_size, type=int)
    parser.add_argument("--train_batch_size", dest="train_batch_size", default=settings.arclength_default_train_batch_size, type=int)
    parser.add_argument("--validation_batch_size", dest="validation_batch_size", default=settings.arclength_default_validation_batch_size, type=int)
    parser.add_argument("--train_dataset_size", dest="train_dataset_size", default=settings.arclength_default_train_dataset_size, type=int)
    parser.add_argument("--validation_dataset_size", dest="validation_dataset_size", default=settings.arclength_default_validation_dataset_size, type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", default=settings.arclength_default_learning_rate, type=float)
    parser.add_argument("--validation_split", dest="validation_split", default=settings.arclength_default_validation_split, type=float)
    parser.add_argument("--supporting_points_count", dest="supporting_points_count", default=settings.arclength_default_supporting_points_count, type=int)
    parser.add_argument("--anchor_points_count", dest="anchor_points_count", default=settings.arclength_default_anchor_points_count, type=int)
    parser.add_argument("--multimodality", dest="multimodality", default=settings.arclength_default_multimodality, type=int)
    parser.add_argument("--min_offset", dest="min_offset", default=settings.arclength_default_min_offset, type=int)
    parser.add_argument("--max_offset", dest="max_offset", default=settings.arclength_default_max_offset, type=int)
    parser.add_argument("--num_workers_train", dest="num_workers_train", default=settings.arclength_default_num_workers_train, type=int)
    parser.add_argument("--num_workers_validation", dest="num_workers_validation", default=settings.arclength_default_num_workers_validation, type=int)
    parser.add_argument("--history_size", dest="history_size", default=settings.arclength_default_history_size, type=int)
    args = parser.parse_args()

    OnlineDataset = None
    results_base_dir_path = None
    if args.group == 'euclidean':
        OnlineDataset = DeepSignatureEuclideanArclengthTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_euclidean_arclength_tuplets_results_dir_path
    elif args.group == 'equiaffine':
        OnlineDataset = DeepSignatureEquiaffineArclengthTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_equiaffine_arclength_tuplets_results_dir_path
    elif args.group == 'affine':
        OnlineDataset = DeepSignatureAffineArclengthTupletsOnlineDataset
        results_base_dir_path = settings.level_curves_affine_arclength_tuplets_results_dir_path

    train_dataset = OnlineDataset(
        dataset_size=args.train_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
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
        dir_path=settings.level_curves_dir_path_validation,
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

    model = DeepSignatureArcLengthNet(sample_points=args.supporting_points_count, transformation_group_type=args.group).cuda()
    print(model)

    if args.continue_training:
        latest_subdir = common_utils.get_latest_subdirectory(results_base_dir_path)
        results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
        model.load_state_dict(torch.load(results['model_file_path'], map_location=torch.device('cuda')))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate, line_search_fn='strong_wolfe', history_size=args.history_size)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ArcLengthLoss(anchor_points_count=args.anchor_points_count)
    model_trainer = ModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        validation_split=args.validation_split,
        results_base_dir_path=results_base_dir_path)
