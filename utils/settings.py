# general
data_dir = "C:/deep-signature-data"

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
arclength_default_train_batch_size = int(arclength_default_train_buffer_size / 4)
arclength_default_validation_batch_size = int(arclength_default_validation_buffer_size / 4)
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
