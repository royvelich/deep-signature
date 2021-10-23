import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineCurvatureTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureCurvatureNet
from deep_signature.nn.losses import TupletLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils


if __name__ == '__main__':
    epochs = 10000
    batch_size = 500
    buffer_size = batch_size
    train_dataset_size = batch_size*4
    validation_dataset_size = batch_size
    learning_rate = 1
    validation_split = .2
    supporting_points_count = 3
    sample_points = 2 * supporting_points_count + 1
    sampling_ratio = 0.3
    multimodality = 50
    offset_length = 50
    num_workers = 2

    torch.set_default_dtype(torch.float64)

    train_dataset = DeepSignatureEquiaffineCurvatureTupletsOnlineDataset(
        dataset_size=train_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        replace=True,
        buffer_size=buffer_size,
        num_workers=num_workers,
        supporting_points_count=supporting_points_count,
        offset_length=offset_length)

    validation_dataset = DeepSignatureEquiaffineCurvatureTupletsOnlineDataset(
        dataset_size=validation_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        replace=False,
        buffer_size=buffer_size,
        num_workers=num_workers,
        supporting_points_count=supporting_points_count,
        offset_length=offset_length)

    train_dataset.start()
    validation_dataset.start()

    model = DeepSignatureCurvatureNet(sample_points=sample_points).cuda()
    print(model)

    # device = torch.device('cuda')
    # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_affine_curvature_tuplets_results_dir_path)
    # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe', history_size=800)
    curvature_loss_fn = TupletLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[curvature_loss_fn], optimizer=optimizer)
    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=None,
        results_base_dir_path=settings.level_curves_equiaffine_curvature_tuplets_results_dir_path)
