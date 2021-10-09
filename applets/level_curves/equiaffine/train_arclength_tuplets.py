# import torch
# import os
# import numpy
# from deep_signature.nn.datasets import DeepSignatureTupletsDataset
# from deep_signature.nn.networks import DeepSignatureArcLengthNet
# from deep_signature.nn.losses import ArcLengthLoss
# from deep_signature.nn.losses import NegativeLoss
# from deep_signature.nn.trainers import ModelTrainer
# from common import settings
# from common import utils as common_utils
#
#
# if __name__ == '__main__':
#     epochs = 3000
#     batch_size = 7000
#     learning_rate = 1e-5
#     validation_split = .1
#
#     torch.set_default_dtype(torch.float64)
#     dataset = DeepSignatureTupletsDataset()
#     dataset.load_dataset(dir_path=settings.level_curves_equiaffine_arclength_tuplets_dir_path)
#     model = DeepSignatureArcLengthNet(sample_points=40).cuda()
#     print(model)
#
#     # device = torch.device('cuda')
#     # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_equiaffine_arclength_tuplets_results_dir_path)
#     # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
#     # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     tuplet_loss_fn = ArcLengthLoss()
#     # negative_loss_fn = NegativeLoss(factor=100)
#     # model_trainer = ModelTrainer(model=model, loss_functions=[tuplet_loss_fn, negative_loss_fn], optimizer=optimizer)
#     model_trainer = ModelTrainer(model=model, loss_functions=[tuplet_loss_fn], optimizer=optimizer)
#     model_trainer.fit(
#         dataset=dataset,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_split=validation_split,
#         results_base_dir_path=settings.level_curves_equiaffine_arclength_tuplets_results_dir_path)


import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureEquiaffineArclengthTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils


if __name__ == '__main__':
    learning_rate = .1
    validation_split = .2
    epochs = 10000
    batch_size = 500
    buffer_size = batch_size
    dataset_size = batch_size * 5
    supporting_points_count = 6
    sampling_ratio = 0.3
    multimodality = 50
    offset_length = 50
    num_workers = 1
    section_points_count = 12

    torch.set_default_dtype(torch.float64)

    dataset = DeepSignatureEquiaffineArclengthTupletsOnlineDataset(
        dataset_size=dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        buffer_size=buffer_size,
        num_workers=num_workers,
        section_points_count=section_points_count)

    dataset.start()

    model = DeepSignatureArcLengthNet(sample_points=section_points_count).cuda()
    print(model)

    # device = torch.device('cuda')
    # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_affine_curvature_tuplets_results_dir_path)
    # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe', history_size=800)
    loss_fn = ArcLengthLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
    model_trainer.fit(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        results_base_dir_path=settings.level_curves_equiaffine_arclength_tuplets_results_dir_path)
