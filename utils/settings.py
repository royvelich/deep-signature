import os


# general
data_dir = "C:/deep-signature-data"
images_dir_path_train = os.path.normpath(os.path.join(data_dir, "images/train"))
images_dir_path_validation = os.path.normpath(os.path.join(data_dir, "images/validation"))
images_dir_path_test = os.path.normpath(os.path.join(data_dir, "images/test"))
plots_dir = os.path.normpath(os.path.join(data_dir, "plots"))

# circles
circles_dir_path_train = os.path.normpath(os.path.join(data_dir, "circles/curves/train"))
circles_dir_path_test = os.path.normpath(os.path.join(data_dir, "circles/curves/test"))
circles_section_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "circles/datasets/tuplets/sections"))
circles_triangle_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "circles/datasets/tuplets/triangles"))
circles_section_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "circles/results/tuplets/sections"))
circles_triangle_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "circles/results/tuplets/triangles"))

# level-curves
level_curves_dir_path_train = os.path.normpath(os.path.join(data_dir, "level-curves/curves/train"))
level_curves_dir_path_validation = os.path.normpath(os.path.join(data_dir, "level-curves/curves/validation"))
level_curves_dir_path_test = os.path.normpath(os.path.join(data_dir, "level-curves/curves/test"))
level_curves_euclidean_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/euclidean"))
level_curves_euclidean_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/euclidean"))
level_curves_equiaffine_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/equiaffine"))
level_curves_equiaffine_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/equiaffine"))
level_curves_similarity_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/similarity"))
level_curves_similarity_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/similarity"))
level_curves_affine_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/affine"))
level_curves_affine_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/affine"))
level_curves_euclidean_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/euclidean"))
level_curves_euclidean_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/euclidean"))
level_curves_similarity_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/similarity"))
level_curves_similarity_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/similarity"))
level_curves_equiaffine_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/equiaffine"))
level_curves_equiaffine_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/equiaffine"))
level_curves_affine_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/affine"))
level_curves_affine_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/affine"))

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
arclength_default_learning_rate = 1
arclength_default_validation_split = None
arclength_default_epochs = None
arclength_default_train_buffer_size = 170000
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
arclength_default_num_workers_train = 5
arclength_default_num_workers_validation = 5
arclength_default_history_size = 2500

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