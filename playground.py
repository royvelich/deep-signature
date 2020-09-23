import os
import scipy.io
import numpy
import re
import scipy.stats as ss
import matplotlib.pyplot as plt
import deep_signature.dist
import deep_signature.data
import gzip


if __name__ == '__main__':

    metadata = numpy.load(file='./dataset/metadata.npy', allow_pickle=True)

    bla = metadata.item()

    curve = numpy.load(file='C:\\Users\\Roy\\Documents\\GitHub\\deep-signature\\dataset\\1245\\4\\2\\sample.npy', allow_pickle=True)

    g = 5

    arr1 = numpy.array([[1,2],[4,6],[7,5]])

    arr1_da = numpy.concatenate((arr1[-2:],arr1,arr1[:2]), axis=0)

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
