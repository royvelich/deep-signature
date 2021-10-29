import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureEuclideanArclengthTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils


if __name__ == '__main__':
    learning_rate = 1
    validation_split = None
    epochs = None
    train_buffer_size = 5000
    validation_buffer_size = 2000
    train_batch_size = 5000
    validation_batch_size = validation_buffer_size
    train_dataset_size = 3*train_batch_size
    validation_dataset_size = validation_batch_size
    sampling_ratio = 0.3
    multimodality = 50
    offset_length = 50
    num_workers = 1
    section_points_count = 10
    supporting_points_count = 20
    min_offset = supporting_points_count
    max_offset = 6*supporting_points_count

    torch.set_default_dtype(torch.float64)

    train_dataset = DeepSignatureEuclideanArclengthTupletsOnlineDataset(
        dataset_size=train_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        replace=True,
        buffer_size=train_buffer_size,
        num_workers=num_workers,
        section_points_count=section_points_count,
        supporting_points_count=supporting_points_count,
        min_offset=min_offset,
        max_offset=max_offset)

    validation_dataset = DeepSignatureEuclideanArclengthTupletsOnlineDataset(
        dataset_size=validation_dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        replace=False,
        buffer_size=train_buffer_size,
        num_workers=num_workers,
        section_points_count=section_points_count,
        supporting_points_count=supporting_points_count,
        min_offset=min_offset,
        max_offset=max_offset)

    validation_dataset.start()
    validation_dataset.stop()
    train_dataset.start()

    model = DeepSignatureArcLengthNet(sample_points=supporting_points_count).cuda()
    print(model)

    # device = torch.device('cuda')
    # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_euclidean_curvature_tuplets_results_dir_path)
    # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe', history_size=300)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ArcLengthLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=epochs,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        validation_split=validation_split,
        results_base_dir_path=settings.level_curves_euclidean_arclength_tuplets_results_dir_path)
