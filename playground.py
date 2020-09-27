import numpy
import torch


if __name__ == '__main__':

    print(torch.__version__)

    t = torch.tensor([[2,1,2],[2,5,2],[1,1,4]], dtype=torch.float32)
    norms = torch.norm(t)


    t1 = torch.tensor([1,0,1,1,1,0,1,0,0,1], dtype=torch.float32)
    t2 = torch.tensor([2,2,2,2,2,2,2,2,2,2], dtype=torch.float32)
    t3 = 1 - t1
    t4 = torch.tensor([2,-5,5,1,2,-1,-0.1,8,-7,6], dtype=torch.float32)

    bla = t1 * t2
    bla2 = torch.sum(bla)

    bla3 = torch.max(torch.zeros_like(t1), t4)




    metadata = numpy.load(file='./dataset/metadata.npy', allow_pickle=True)

    bla = metadata.item()

    curve = numpy.load(file='C:\\Users\\Roy\\Documents\\GitHub\\deep-signature\\dataset\\1268\\0\\8\\sample.npy', allow_pickle=True)

    g = 5

    arr1 = numpy.array([[1,2],[4,6],[7,5]])

    curve_torch = torch.from_numpy(curve)

    print(curve_torch[:, 0])

    # arr1_da = numpy.concatenate((arr1[-2:],arr1,arr1[:2]), axis=0)

    arr2 = numpy.array([[7,3],[9,6],[0,3]])
    #
    # print(numpy.concatenate((arr1, arr2), axis=1))

    # print(arr1)
    bob = numpy.vstack((numpy.transpose(arr1), numpy.transpose(arr2)))
    print(bob)
    print(bob.shape)

    # print(arr[0])
    # print(arr[1:4])
    # print(arr[4:8])
