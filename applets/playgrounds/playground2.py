# class MyClass1:
#     @staticmethod
#     def test(a, b, **kwargs):
#         return MyClass1.test3(**kwargs)
#
#     @staticmethod
#     def test3(bla2, bla1):
#         return 5
#
#     @classmethod
#     def test2(cls):
#         return 6
#
# class MyClass2(MyClass1):
#     @staticmethod
#     def test():
#         return 1
#
#     @staticmethod
#     def test3():
#         return 8

from deep_signature.data_generation import curve_generation
from deep_signature.data_manipulation import curve_sampling
import numpy

if __name__ == '__main__':

    curve = numpy.random.uniform(low=0, high=20, size=[15, 2])
    curve_sampling.sample_curve(curve=curve, max_offset=14, supporting_point_count=1)

    bla = numpy.arange(start=2, stop=20)
    g = 5
    # h = [1,2,3] * 7
    #
    #
    # a = [1,2,3]
    # b = [4,5,6]
    # c = [6,7,8]
    # d = [a,b,c]
    #
    # bla = list(zip(*d))
    # h=6
    #

    # curve_generation.LevelCurvesGenerator.generate_curves(
    #     dir_path="./mydata",
    #     curves_count=1000,
    #     images_base_dir_path="C:/deep-signature-data/images2",
    #     sigmas=[2, 4, 8, 16, 32],
    #     contour_levels=[0.2, 0.5, 0.8],
    #     min_points=1000,
    #     max_points=6000,
    #     flat_point_threshold=1e-3,
    #     max_flat_points_ratio=0.04,
    #     max_abs_kappa=8
    # )

    curve_generation.LevelCurvesGenerator.generate_curves(
        dir_path="./mydata",
        curves_count=100,
        images_base_dir_path="C:/deep-signature-data/images2",
        sigmas=[2],
        contour_levels=[0.2],
        min_points=1000,
        max_points=6000,
        flat_point_threshold=1e-3,
        max_flat_points_ratio=0.04,
        max_abs_kappa=8
    )