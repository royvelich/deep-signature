import numpy
import torch
from deep_signature.data_generation import DatasetGenerator

if __name__ == '__main__':

    dataset_generator = DatasetGenerator()
    dataset_generator.load_raw_curves(dir_path='C:\\Users\\Roy\\OneDrive - Technion\\deep-signature-raw-data\\raw-data-new')
    dataset_generator.generate_curves(
        rotation_factor=12,
        sampling_factor=15,
        sample_points=600)

    # a = numpy.array([[0., 0.], [0.3, 0.], [1.25, -0.1],
    #               [2.1, -0.9], [2.85, -2.3], [3.8, -3.95],
    #               [5., -5.75], [6.4, -7.8], [8.05, -9.9],
    #               [9.9, -11.6], [12.05, -12.85], [14.25, -13.7],
    #               [16.5, -13.8], [19.25, -13.35], [21.3, -12.2],
    #               [22.8, -10.5], [23.55, -8.15], [22.95, -6.1],
    #               [21.35, -3.95], [19.1, -1.9]])
    #
    # dx_dt = numpy.gradient(a[:, 0])
    # dy_dt = numpy.gradient(a[:, 1])
    # bla = numpy.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    #
    # bla2 = numpy.gradient(a, axis=0)
    #
    # h = 5


    # print(torch.__version__)
    #
    # t = torch.tensor([[2,1,2],[2,5,2],[1,1,4]], dtype=torch.float32)
    # norms = torch.norm(t)
    #
    #
    # t1 = torch.tensor([1,0,1,1,1,0,1,0,0,1], dtype=torch.float32)
    # t2 = torch.tensor([2,2,2,2,2,2,2,2,2,2], dtype=torch.float32)
    # t3 = 1 - t1
    # t4 = torch.tensor([2,-5,5,1,2,-1,-0.1,8,-7,6], dtype=torch.float32)
    #
    # bla = t1 * t2
    # bla2 = torch.sum(bla)
    #
    # bla3 = torch.max(torch.zeros_like(t1), t4)
    #
    #
    #
    #
    # metadata = numpy.load(file='./dataset/metadata.npy', allow_pickle=True)
    #
    # bla = metadata.item()
    #
    # curve = numpy.load(file='C:\\Users\\Roy\\Documents\\GitHub\\deep-signature\\dataset\\1268\\0\\8\\sample.npy', allow_pickle=True)
    #
    # g = 5
    #
    # arr1 = numpy.array([[1,2],[4,6],[7,5]])
    #
    # curve_torch = torch.from_numpy(curve)
    #
    # print(curve_torch[:, 0])
    #
    # # arr1_da = numpy.concatenate((arr1[-2:],arr1,arr1[:2]), axis=0)
    #
    # arr2 = numpy.array([[7,3],[9,6],[0,3]])
    # #
    # # print(numpy.concatenate((arr1, arr2), axis=1))
    #
    # # print(arr1)
    # bob = numpy.vstack((numpy.transpose(arr1), numpy.transpose(arr2)))
    # print(bob)
    # print(bob.shape)
    #
    # # print(arr[0])
    # # print(arr[1:4])
    # # print(arr[4:8])
