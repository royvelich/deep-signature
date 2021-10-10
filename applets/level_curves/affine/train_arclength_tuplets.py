# import torch
# import numpy
# from deep_signature.nn.datasets import DeepSignatureTupletsDataset
# from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
# from deep_signature.nn.networks import DeepSignatureArcLengthNet
# from deep_signature.nn.losses import ArcLengthLoss
# from deep_signature.nn.losses import CurvatureLoss
# from deep_signature.nn.trainers import ModelTrainer
# from common import settings
# from common import utils as common_utils
#
#
# if __name__ == '__main__':
#     epochs = 10000
#     batch_size = 150000
#     buffer_size = batch_size
#     dataset_size = batch_size*2
#     learning_rate = 1
#     validation_split = .1
#     supporting_points_count = 6
#     sample_points = 2 * supporting_points_count + 1
#     sampling_ratio = 0.3
#     multimodality = 50
#     offset_length = 50
#     num_workers = 1
#
#     torch.set_default_dtype(torch.float64)
#
#     dataset = DeepSignatureAffineArclengthTupletsOnlineDataset(
#         dataset_size=dataset_size,
#         dir_path=settings.level_curves_dir_path_train,
#         sampling_ratio=sampling_ratio,
#         multimodality=multimodality,
#         supporting_points_count=supporting_points_count,
#         buffer_size=buffer_size,
#         num_workers=num_workers)
#
#     dataset.start()
#
#     model = DeepSignatureArcLengthNet(sample_points=sample_points).cuda()
#     print(model)
#
#     # device = torch.device('cuda')
#     # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_affine_curvature_tuplets_results_dir_path)
#     # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
#     # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))
#
#     optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe', history_size=800)
#     loss_fn = ArcLengthLoss()
#     model_trainer = ModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
#     model_trainer.fit(
#         dataset=dataset,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_split=validation_split,
#         results_base_dir_path=settings.level_curves_affine_curvature_tuplets_results_dir_path)


import torch
import numpy
from deep_signature.nn.datasets import DeepSignatureTupletsDataset
from deep_signature.nn.datasets import DeepSignatureAffineArclengthTupletsOnlineDataset
from deep_signature.nn.networks import DeepSignatureArcLengthNet
from deep_signature.nn.losses import ArcLengthLoss
from deep_signature.nn.losses import CurvatureLoss
from deep_signature.nn.trainers import ModelTrainer
from common import settings
from common import utils as common_utils


if __name__ == '__main__':
    learning_rate = .1
    validation_split = .2
    epochs = 10000000
    batch_size = 1000
    buffer_size = batch_size
    dataset_size = batch_size * 5
    sampling_ratio = 0.3
    multimodality = 50
    offset_length = 50
    num_workers = 4
    section_points_count = 10
    supporting_points_count = 40
    min_offset = 40
    max_offset = 40

    torch.set_default_dtype(torch.float64)

    dataset = DeepSignatureAffineArclengthTupletsOnlineDataset(
        dataset_size=dataset_size,
        dir_path=settings.level_curves_dir_path_train,
        sampling_ratio=sampling_ratio,
        multimodality=multimodality,
        buffer_size=buffer_size,
        num_workers=num_workers,
        section_points_count=section_points_count,
        supporting_points_count=supporting_points_count,
        min_offset=min_offset,
        max_offset=max_offset)

    dataset.start()

    model = DeepSignatureArcLengthNet(sample_points=supporting_points_count).cuda()
    print(model)

    # device = torch.device('cuda')
    # latest_subdir = common_utils.get_latest_subdirectory(settings.level_curves_affine_curvature_tuplets_results_dir_path)
    # results = numpy.load(f"{latest_subdir}/results.npy", allow_pickle=True).item()
    # model.load_state_dict(torch.load(results['model_file_path'], map_location=device))

    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe', history_size=300)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ArcLengthLoss()
    model_trainer = ModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
    model_trainer.fit(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        results_base_dir_path=settings.level_curves_affine_arclength_tuplets_results_dir_path)
