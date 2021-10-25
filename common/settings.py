import os


# general
data_dir = "C:/deep-signature-data"
images_dir_path_train = os.path.normpath(os.path.join(data_dir, "images/train"))
images_dir_path_test = os.path.normpath(os.path.join(data_dir, "images/test"))

# circles
circles_dir_path_train = os.path.normpath(os.path.join(data_dir, "circles/curves/train"))
circles_dir_path_test = os.path.normpath(os.path.join(data_dir, "circles/curves/test"))
circles_section_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "circles/datasets/tuplets/sections"))
circles_triangle_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "circles/datasets/tuplets/triangles"))
circles_section_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "circles/results/tuplets/sections"))
circles_triangle_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "circles/results/tuplets/triangles"))

# level-curves
level_curves_dir_path_train = os.path.normpath(os.path.join(data_dir, "level-curves/curves/train"))
level_curves_dir_path_test = os.path.normpath(os.path.join(data_dir, "level-curves/curves/test"))
level_curves_euclidean_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/euclidean"))
level_curves_euclidean_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/euclidean"))
level_curves_equiaffine_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/equiaffine"))
level_curves_equiaffine_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/equiaffine"))
level_curves_affine_curvature_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/curvature/affine"))
level_curves_affine_curvature_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/curvature/affine"))
level_curves_euclidean_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/euclidean"))
level_curves_euclidean_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/euclidean"))
level_curves_equiaffine_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/equiaffine"))
level_curves_equiaffine_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/equiaffine"))
level_curves_affine_arclength_tuplets_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/datasets/tuplets/arclength/affine"))
level_curves_affine_arclength_tuplets_results_dir_path = os.path.normpath(os.path.join(data_dir, "level-curves/results/tuplets/arclength/affine"))
