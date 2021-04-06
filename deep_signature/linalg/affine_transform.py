# numpy
import numpy


def random_equiaffine_transform_2d(max_scale=2, min_eig_value_ratio=1.5, max_eig_value_ratio=6):
    while True:
        scale = numpy.random.uniform(low=0, high=max_scale, size=2)
        coeffs = numpy.random.random(size=2)
        entries = scale * coeffs
        L = numpy.array([[1, 0], [entries[0], 1]])
        U = numpy.array([[1, entries[1]], [0, 1]])
        A = numpy.matmul(L, U)
        w, v = numpy.linalg.eig(A)
        r1 = w[0] / w[1]
        r2 = w[1] / w[0]
        r = numpy.maximum(r1, r2)
        if (r > min_eig_value_ratio) and (r < max_eig_value_ratio):
            return A

    # u, s, vh = numpy.linalg.svd(A, full_matrices=True)
    # bla = 5
    # B = numpy.matmul(u, numpy.matmul(numpy.diag(s), vh))
    # return numpy.matmul(L, U)
