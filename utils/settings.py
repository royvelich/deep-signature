import os


# general
data_dir = "C:/deep-signature-data"
images_dir_path_train = os.path.normpath(os.path.join(data_dir, "images/train"))
images_dir_path_validation = os.path.normpath(os.path.join(data_dir, "images/validation"))
images_dir_path_test = os.path.normpath(os.path.join(data_dir, "images/test"))
plots_dir = os.path.normpath(os.path.join(data_dir, "plots"))

# level-curves
level_curves_dir_path_train = os.path.normpath(os.path.join(data_dir, "curves/train"))
level_curves_dir_path_validation = os.path.normpath(os.path.join(data_dir, "curves/validation"))
level_curves_dir_path_test = os.path.normpath(os.path.join(data_dir, "curves/test"))

dataset_dir_path_template = os.path.normpath(os.path.join(data_dir, "datasets/%s/%s/%s"))
results_dir_path_template = os.path.normpath(os.path.join(data_dir, "results/%s/%s"))

# equiaffine_curvature_train_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/equiaffine/train"))
# equiaffine_curvature_validation_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/equiaffine/validation"))
# equiaffine_curvature_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/curvature/equiaffine"))
#
# similarity_curvature_train_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/similarity/train"))
# similarity_curvature_validation_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/similarity/validation"))
# similarity_curvature_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/curvature/similarity"))
#
# affine_curvature_train_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/affine/train"))
# affine_curvature_validation_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/curvature/affine/validation"))
# affine_curvature_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/curvature/affine"))
#
# euclidean_arclength_train_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/euclidean/train"))
# euclidean_arclength_validation_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/euclidean"))
# euclidean_arclength_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/arclength/euclidean"))
#
# similarity_arclength_train_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/similarity/train"))
# similarity_arclength_validation_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/similarity"))
# similarity_arclength_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/arclength/similarity"))
#
# equiaffine_arclength_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/equiaffine/train"))
# equiaffine_arclength_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/equiaffine/validation"))
# equiaffine_arclength_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/arclength/equiaffine"))
#
# affine_arclength_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/affine"))
# affine_arclength_dataset_dir_path = os.path.normpath(os.path.join(data_dir, "datasets/arclength/affine"))
# affine_arclength_results_dir_path = os.path.normpath(os.path.join(data_dir, "results/arclength/affine"))

# curvature
curvature_default_continue_training = False
curvature_default_epochs = None
curvature_default_train_buffer_size = 250000
curvature_default_validation_buffer_size = 50000
curvature_default_train_batch_size = curvature_default_train_buffer_size
curvature_default_validation_batch_size = curvature_default_validation_buffer_size
curvature_default_train_dataset_size = curvature_default_train_batch_size
curvature_default_validation_dataset_size = curvature_default_validation_batch_size
curvature_default_learning_rate = 1
curvature_default_validation_split = None
curvature_default_supporting_points_count = 3
curvature_default_sample_points_count = 2 * curvature_default_supporting_points_count + 1
curvature_default_sampling_ratio = 0.5
curvature_default_multimodality = 30
curvature_default_offset_length = 50
curvature_default_num_workers_train = 10
curvature_default_num_workers_validation = 10
curvature_default_negative_examples_count = 3
curvature_default_history_size = 1500

# arclength
arclength_default_continue_training = False
arclength_default_learning_rate = 0.1
arclength_default_validation_split = None
arclength_default_epochs = None
arclength_default_train_buffer_size = 100000
arclength_default_validation_buffer_size = 50000
arclength_default_train_batch_size = arclength_default_train_buffer_size
arclength_default_validation_batch_size = arclength_default_validation_buffer_size
arclength_default_train_dataset_size = arclength_default_train_batch_size
arclength_default_validation_dataset_size = arclength_default_validation_batch_size
arclength_default_multimodality = 5
arclength_default_supporting_points_count = 5
arclength_default_min_offset = arclength_default_supporting_points_count - 1
arclength_default_max_offset = 2 * arclength_default_min_offset
arclength_default_anchor_points_count = 5
arclength_default_num_workers_train = 6
arclength_default_num_workers_validation = 6
arclength_default_history_size = 2500

equiaffine_arclength_min_cond_training = 1.3
equiaffine_arclength_max_cond_training = 2
affine_arclength_min_cond_training = 1.3
affine_arclength_max_cond_training = 2
affine_arclength_min_det_training = 1.3
affine_arclength_max_det_training = 2

equiaffine_arclength_min_cond_evaluation = 1.3
equiaffine_arclength_max_cond_evaluation = 2
affine_arclength_min_cond_evaluation = 1.5
affine_arclength_max_cond_evaluation = 2
affine_arclength_min_det_evaluation = 1.1
affine_arclength_max_det_evaluation = 1.6

# plots
matplotlib_factor = 1
matplotlib_axis_title_label_fontsize = int(35 * matplotlib_factor)
matplotlib_axis_tick_label_fontsize = int(25 * matplotlib_factor)
matplotlib_fig_title_label_fontsize = int(25 * matplotlib_factor)
matplotlib_legend_label_fontsize = int(25 * matplotlib_factor)
matplotlib_sample_point_size = int(20 * matplotlib_factor)
matplotlib_sample_anchor_size = int(40 * matplotlib_factor)
matplotlib_line_point_size = int(2 * matplotlib_factor)
matplotlib_graph_line_width = 6
matplotlib_figsize = (60, 20)

plotly_fontsize_factor = 1.5
plotly_pointsize_factor = 1.5
plotly_axis_title_label_fontsize = int(25 * plotly_fontsize_factor)
plotly_axis_tick_label_fontsize = int(25 * plotly_fontsize_factor)
plotly_fig_title_label_fontsize = int(35 * plotly_fontsize_factor)
plotly_legend_label_fontsize = int(25 * plotly_fontsize_factor)
plotly_sample_point_size = int(6 * plotly_pointsize_factor)
plotly_sample_anchor_size = int(10 * plotly_pointsize_factor)
plotly_line_point_size = int(12 * plotly_pointsize_factor)
plotly_graph_line_width = 4
plotly_write_image_width = 3500
plotly_write_image_height = 1000
