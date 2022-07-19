import os

# general
data_dir = "C:/deep-signature-data"
plots_dir = os.path.normpath(os.path.join(data_dir, "plots"))

default_continue_training = False
default_epochs = None
default_train_buffer_size = 20000
default_validation_buffer_size = 10000
default_train_batch_size = default_train_buffer_size
default_validation_batch_size = default_validation_buffer_size
default_train_dataset_size = default_train_batch_size
default_validation_dataset_size = default_validation_batch_size
default_learning_rate = 0.001
default_validation_split = None
default_supporting_points_count = 3
default_sample_points_count = 2 * default_supporting_points_count + 1
default_sampling_ratio = 0.3
default_multimodality = 30
default_multimodality_evaluation = 30
default_offset_length = 30
default_num_workers_train = 5
default_num_workers_validation = 5
default_negative_examples_count = 2
default_history_size = 800
default_min_offset = default_supporting_points_count - 1
default_max_offset = 2 * default_min_offset
default_anchor_points_count = 4

equiaffine_arclength_min_cond_training = 1
equiaffine_arclength_max_cond_training = 6
affine_arclength_min_cond_training = 1
affine_arclength_max_cond_training = 4
affine_arclength_min_det_training = 1
affine_arclength_max_det_training = 4

equiaffine_arclength_min_cond_evaluation = 3
equiaffine_arclength_max_cond_evaluation = 3
affine_arclength_min_cond_evaluation = 2
affine_arclength_max_cond_evaluation = 2
affine_arclength_min_det_evaluation = 2
affine_arclength_max_det_evaluation = 2

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
