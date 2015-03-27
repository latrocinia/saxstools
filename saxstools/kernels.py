from __future__ import print_function
import numpy as np
import os.path
import pyopencl as cl

class Kernels():


    def __init__(self, ctx):
        self.context = ctx

        self.kernel_file = os.path.join(os.path.dirname(__file__), 'kernels.cl')
        self.kernels = cl.Program(ctx, open(self.kernel_file).read()).build()


    def calc_chi2(self, queue, interspace, q, Iq, 
            rind, rxyz, lind, lxyz, origin, voxelspacing, fifj, targetIq, sq, chi2):

        kernel = self.kernels.calc_chi2
        workgroupsize = 16

        gws = (queue.device.max_compute_units * workgroupsize * 1024,)
        lws = (workgroupsize,)

        floatsize = 4
        tmpIq = cl.LocalMemory(floatsize * q.shape[0] * workgroupsize)

        shape = np.zeros(4, dtype=np.int32)
        shape[:-1] = interspace.shape
        shape[-1] = interspace.size

        nq = np.int32(q.shape[0])
        nind1 = np.int32(rind.shape[0])
        nind2 = np.int32(lind.shape[0])

        fifj_shape = np.zeros(4, dtype=np.int32)
        fifj_shape[:-1] = fifj.shape
        fifj_shape[-1] = fifj.size

        kernel.set_args(interspace.data, q.data, Iq.data, tmpIq, rind.data, rxyz.data,
                lind.data, lxyz.data, origin, voxelspacing, fifj.data, targetIq.data, sq.data, chi2.data,
                shape, nq, nind1, nind2, fifj_shape)
        status = cl.enqueue_nd_range_kernel(queue, kernel, gws, lws)

        return status

    def rotate_points(self, queue, xyz, rotmat, rot_xyz):

        kernel = self.kernels.rotate_points

        npoints = np.int32(xyz.shape[0])
        rotmat16 = np.zeros(16, dtype=np.float32)
        rotmat16[:9] = rotmat.ravel()

        kernel.set_args(rotmat16, xyz.data, rot_xyz.data, npoints)

        gws = (int(npoints), )

        status = cl.enqueue_nd_range_kernel(queue, kernel, gws, None)

        return status
