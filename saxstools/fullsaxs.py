from __future__ import print_function, absolute_import, division
from sys import stdout as _stdout
from time import time as _time
from math import ceil

import numpy as np

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    irfftn = pyfftw.interfaces.numpy_fft.irfftn

except ImportError:
    from numpy.fft import rfftn, irfftn


from disvis import volume
from disvis.points import dilate_points
from disvis.libdisvis import (rotate_image3d, dilate_points_add, longest_distance)
from saxstools.saxs_curve import scattering_curve, create_fifj_lookup_table
from saxstools.helpers import coarse_grain
from saxstools.libsaxstools import calc_chi2

        
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    import disvis.pyclfft
    from disvis.kernels import Kernels
    from disvis import pyclfft
except ImportError:
    pass

class FullSAXS(object):

    def __init__(self):
        # parameters to be defined
        self._receptor = None
        self._ligand = None

        # parameters with standard values
        self.rotations = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        self.weights = None
        self.voxelspacing = 1.0
        self.interaction_radius = 2.5
        self.max_clash = 100
        self.min_interaction = 300

        self.coarse_grain = True
        self.beads_per_residue = 2

        # CPU or GPU
        self._queue = None

        # unchangeable
        self._data = {}
        self._q = None
        self._Iq = None
        self._sq = None

    @property
    def receptor(self):
        return self._receptor

    @receptor.setter
    def receptor(self, receptor):
        self._receptor = receptor.duplicate()

    @property
    def ligand(self):
        return self._ligand

    @ligand.setter
    def ligand(self, ligand):
        self._ligand = ligand.duplicate()

    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, rotations):
        rotmat = np.asarray(rotations, dtype=np.float64)

        if rotmat.ndim != 3:
            raise ValueError("Input should be a list of rotation matrices.")

        self._rotations = rotmat

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def interaction_radius(self):
        return self._interaction_radius

    @interaction_radius.setter
    def interaction_radius(self, radius):
        if radius <= 0:
            raise ValueError("Interaction radius should be bigger than zero")
        self._interaction_radius = radius

    @property
    def voxelspacing(self):
        return self._voxelspacing

    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def max_clash(self):
        return self._max_clash

    @max_clash.setter
    def max_clash(self, max_clash):
        if max_clash < 0:
            raise ValueError("Maximum allowed clashing volume cannot be negative")
        self._max_clash = max_clash + 0.9

    @property
    def min_interaction(self):
        return self._min_interaction

    @min_interaction.setter
    def min_interaction(self, min_interaction):
        if min_interaction < 1:
            raise ValueError("Minimum required interaction volume cannot be smaller than 1")
        self._min_interaction = min_interaction + 0.9
        
    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue

    @property
    def data(self):
        return self._data


    @property
    def saxsdata(self):
        return self._q, self._Iq, self._sq


    @saxsdata.setter
    def saxsdata(self, saxsdata):
        self._q, self._Iq, self._sq = saxsdata


    def _initialize(self):

        # check if requirements are set
        if any(x is None for x in (self.receptor, self.ligand)):
            raise ValueError("Not all requirements are met for a search")

        if self.weights is None:
            self.weights = np.ones(self.rotations.shape[0], dtype=np.float64)

        if len(self.weights) != len(self.rotations):
            raise ValueError("")

        d = self.data

        # determine size for grid
        shape = grid_shape(self.receptor.coor, self.ligand.coor, self.voxelspacing)

        # calculate the interaction surface and core of the receptor
        vdw_radii = self.receptor.vdw_radius
        radii = vdw_radii + self.interaction_radius
        d['rsurf'] = rsurface(self.receptor.coor, radii, 
                shape, self.voxelspacing)
        d['rcore'] = rsurface(self.receptor.coor, vdw_radii, 
                shape, self.voxelspacing)

        # keep track of some data for later calculations
        d['origin'] = np.asarray(d['rcore'].origin, dtype=np.float64)
        d['shape'] = d['rcore'].shape
        d['start'] = d['rcore'].start
        d['nrot'] = self.rotations.shape[0]

        # set ligand center to the origin of the receptor map
        # and make a grid of the ligand
        radii = self.ligand.vdw_radius
        d['lsurf'] = dilate_points((self.ligand.coor - self.ligand.center \
                + self.receptor.center), radii, volume.zeros_like(d['rcore']))
        d['im_center'] = np.asarray((self.receptor.center - d['rcore'].origin)/self.voxelspacing, dtype=np.float64)

        d['max_clash'] = self.max_clash/self.voxelspacing**3
        d['min_interaction'] = self.min_interaction/self.voxelspacing**3

        # SAXS data
	d['q'] = self._q
        d['targetIq'] = self._Iq
	d['sq'] = self._sq

        if self.coarse_grain:
            e1, xyz1 = coarse_grain(self.receptor, bpr=self.beads_per_residue)
            e2, xyz2 = coarse_grain(self.ligand, bpr=self.beads_per_residue)

        else:
            e1, xyz1 = self.receptor.elements, self.receptor.coor
            e2, xyz2 = self.ligand.elements, self.ligand.coor

        d['base_Iq'] = scattering_curve(self._q, e1, xyz1, bpr=self.beads_per_residue)
        d['base_Iq'] += scattering_curve(self._q, e2, xyz2, bpr=self.beads_per_residue)

        d['fifj'], d['rind'], d['lind'] = create_fifj_lookup_table(d['q'], e1, e2, bpr=self.beads_per_residue)
        d['rxyz'] = xyz1
        d['lxyz'] = xyz2 - self.ligand.center

        d['chi2'] = np.zeros(d['rcore'].shape, dtype=np.float64)
        d['best_chi2'] = np.zeros_like(d['chi2'])


    def search(self):
        self._initialize()
        if self.queue is None:
            self._cpu_init()
            self._cpu_search()
        else:
            self._gpu_init()
            self._gpu_search()

        if _stdout.isatty():
            print()
        
        d = self.data
        ind = d['best_chi2'] > 0
        d['best_chi2'][ind] -= d['best_chi2'][ind].min()
	return volume.Volume(d['best_chi2'], voxelspacing=self.voxelspacing, origin=d['origin'])

    def _cpu_init(self):

        self.cpu_data = {}
        c = self.cpu_data
        d = self.data

        c['rcore'] = d['rcore'].array
        c['rsurf'] = d['rsurf'].array
        c['im_lsurf'] = d['lsurf'].array

        c['lsurf'] = np.zeros_like(c['rcore'])
        c['clashvol'] = np.zeros_like(c['rcore'])
        c['intervol'] = np.zeros_like(c['rcore'])
        c['interspace'] = np.zeros_like(c['rcore'], dtype=np.int64)

        # complex arrays
        c['ft_shape'] = list(d['shape'])
        c['ft_shape'][-1] = d['shape'][-1]//2 + 1
        c['ft_lsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rcore'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)

        # initial calculations
        c['ft_rcore'] = rfftn(c['rcore'])
        c['ft_rsurf'] = rfftn(c['rsurf'])
        c['rotmat'] = np.asarray(self.rotations, dtype=np.float64)
        c['weights'] = np.asarray(self.weights, dtype=np.float64)

        c['nrot'] = d['nrot']
        c['shape'] = d['shape']
        c['max_clash'] = d['max_clash']
        c['min_interaction'] = d['min_interaction']
        c['vlength'] = int(np.linalg.norm(self.ligand.coor - \
                self.ligand.center, axis=1).max() + \
                self.interaction_radius + 1.5)/self.voxelspacing
        c['origin'] = d['origin']

        # SAXS arrays
	c['q'] = d['q']
        c['targetIq'] = d['targetIq']
	c['sq'] = d['sq']
        c['base_Iq'] = d['base_Iq']
        c['fifj'] = d['fifj']
        c['rind'] = d['rind']
        c['lind'] = d['lind']
        c['rxyz'] = d['rxyz']
        c['lxyz'] = d['lxyz']

        c['chi2'] = d['chi2']
        c['best_chi2'] = d['best_chi2']

        c['Iq'] = np.zeros_like(c['targetIq'])
        c['tmplxyz'] = np.zeros_like(c['lxyz'])

    def _cpu_search(self):

        d = self.data
        c = self.cpu_data

        time0 = _time()
        for n in xrange(c['rotmat'].shape[0]):
            # rotate ligand image
            rotate_image3d(c['im_lsurf'], c['vlength'], 
                    np.linalg.inv(c['rotmat'][n]), d['im_center'], c['lsurf'])

            c['ft_lsurf'] = rfftn(c['lsurf']).conj()
            c['clashvol'] = irfftn(c['ft_lsurf'] * c['ft_rcore'], s=c['shape'])
            c['intervol'] = irfftn(c['ft_lsurf'] * c['ft_rsurf'], s=c['shape'])

            np.logical_and(c['clashvol'] < c['max_clash'],
                           c['intervol'] > c['min_interaction'],
                           c['interspace'])


	    print('Number of complexes to analyze: ', c['interspace'].sum())
            c['chi2'].fill(0)
            calc_chi2(c['interspace'], c['q'], c['base_Iq'], 
                    c['rind'], c['rxyz'], c['lind'], (np.mat(c['rotmat'][n])*np.mat(c['lxyz']).T).T, 
                    c['origin'], self.voxelspacing, 
                    c['fifj'], c['targetIq'], c['sq'], c['chi2'])

	    np.maximum(c['chi2'], c['best_chi2'], c['best_chi2'])
            if _stdout.isatty():
                self._print_progress(n, c['nrot'], time0)

        d['best_chi2'] = c['best_chi2']

    def _print_progress(self, n, total, time0):
        m = n + 1
        pdone = m/total
        t = _time() - time0
        _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)    '\
                .format(m, total, pdone, 
                        int(t/pdone - t)))
        _stdout.flush()


    def _gpu_init(self):

        self.gpu_data = {}
        g = self.gpu_data
        d = self.data
        q = self.queue

        g['rcore'] = cl_array.to_device(q, float32array(d['rcore'].array))
        g['rsurf'] = cl_array.to_device(q, float32array(d['rsurf'].array))
        g['im_lsurf'] = cl.image_from_array(q.context, float32array(d['lsurf'].array))
        g['sampler'] = cl.Sampler(q.context, False, cl.addressing_mode.CLAMP,
                                  cl.filter_mode.LINEAR)

        g['lsurf'] = cl_array.zeros_like(g['rcore'])
        g['clashvol'] = cl_array.zeros_like(g['rcore'])
        g['intervol'] = cl_array.zeros_like(g['rcore'])
        g['interspace'] = cl_array.zeros(q, d['shape'], dtype=np.int32)

        # complex arrays
        g['ft_shape'] = list(d['shape'])
        g['ft_shape'][0] = d['shape'][0]//2 + 1
        g['ft_rcore'] = cl_array.zeros(q, g['ft_shape'], dtype=np.complex64)
        g['ft_rsurf'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_lsurf'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_clashvol'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_intervol'] = cl_array.zeros_like(g['ft_rcore'])

        # kernels
        g['k'] = Kernels(q.context)
        g['k'].rfftn = pyclfft.RFFTn(q.context, d['shape'])
        g['k'].irfftn = pyclfft.iRFFTn(q.context, d['shape'])

        g['k'].rfftn(q, g['rcore'], g['ft_rcore'])
        g['k'].rfftn(q, g['rsurf'], g['ft_rsurf'])

        g['nrot'] = d['nrot']
        g['max_clash'] = d['max_clash']
        g['min_interaction'] = d['min_interaction']


    def _gpu_search(self):
        d = self.data
        g = self.gpu_data
        q = self.queue
        k = g['k']

        time0 = _time()
        for n in xrange(g['nrot']):

            k.rotate_image3d(q, g['sampler'], g['im_lsurf'],
                    self.rotations[n], g['lsurf'], d['im_center'])

            k.rfftn(q, g['lsurf'], g['ft_lsurf'])
            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rcore'], g['ft_clashvol'])
            k.irfftn(q, g['ft_clashvol'], g['clashvol'])

            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rsurf'], g['ft_intervol'])
            k.irfftn(q, g['ft_intervol'], g['intervol'])

            k.touch(q, g['clashvol'], g['max_clash'],
                    g['intervol'], g['min_interaction'],
                    g['interspace'])

            if _stdout.isatty():
                self._print_progress(n, g['nrot'], time0)

        self.queue.finish()


def rsurface(points, radius, shape, voxelspacing):

    dimensions = [x*voxelspacing for x in shape]
    origin = volume_origin(points, dimensions)
    rsurf = volume.zeros(shape, voxelspacing, origin)

    rsurf = dilate_points(points, radius, rsurf)

    return rsurf


def volume_origin(points, dimensions):

    center = points.mean(axis=0)
    origin = [(c - d/2.0) for c, d in zip(center, dimensions)]

    return origin
    

def grid_restraints(restraints, voxelspacing, origin, lcenter):

    nrestraints = len(restraints)
    g_restraints = np.zeros((nrestraints, 8), dtype=np.float64)

    for n in range(nrestraints):
        r_sel, l_sel, mindis, maxdis = restraints[n]

        r_pos = (r_sel.center - origin)/voxelspacing
        l_pos = (l_sel.center - lcenter)/voxelspacing

        g_restraints[n, 0:3] = r_pos
        g_restraints[n, 3:6] = l_pos
        g_restraints[n, 6] = mindis/voxelspacing
        g_restraints[n, 7] = maxdis/voxelspacing

    return g_restraints


def grid_shape(points1, points2, voxelspacing):
    shape = min_grid_shape(points1, points2, voxelspacing)
    shape = [volume.radix235(x) for x in shape]
    return shape


def min_grid_shape(points1, points2, voxelspacing):
    # the minimal grid shape is the size of the fixed protein in 
    # each dimension and the longest diameter is the scanning chain
    dimensions1 = points1.ptp(axis=0)
    dimension2 = longest_distance(points2)

    grid_shape = np.asarray(((dimensions1 + dimension2)/voxelspacing) + 10, dtype=np.int32)[::-1]

    return grid_shape


def float32array(array_like):
    return np.asarray(array_like, dtype=np.float32)
