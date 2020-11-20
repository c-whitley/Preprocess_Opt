import numpy as np
import pandas as pd 
import ray

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

import multiprocessing as mp

from sklearn.base import BaseEstimator, TransformerMixin


class Rubber_Band( BaseEstimator, TransformerMixin ):
    """
    Applies a rubber band correction to the input matrix of spectra.
    Must be supplied as a shape (n_samples, n_wavenumbers)
    """

    def __init__(self, n_jobs = 1):

        self.n_jobs = n_jobs
   

    def transform(self, X, y=None):

        return self.X - self.baseline

    @ray.remote
    def rubberband_baseline_ray(spectrum):

        wn = np.arange(len(spectrum))
        points = np.column_stack([wn, spectrum])

        verts = ConvexHull(points).vertices

        # Rotate convex hull vertices until they start from the lowest one
        verts = np.roll(verts, -verts.argmin())
        # Leave only the ascending part
        verts = verts[:verts.argmax()]

        baseline = np.interp(wn, wn[verts], spectrum[verts])

        return baseline

    def rubberband_baseline(spectrum):

        wn = np.arange(len(spectrum))
        points = np.column_stack([wn, spectrum])

        verts = ConvexHull(points).vertices

        # Rotate convex hull vertices until they start from the lowest one
        verts = np.roll(verts, -verts.argmin())
        # Leave only the ascending part
        verts = verts[:verts.argmax()]

        baseline = np.interp(wn, wn[verts], spectrum[verts])

        return baseline


    def fit(self, X, y=None):
        
        self.X=X

        if self.n_jobs == 1:

            self.baseline = np.apply_along_axis(self.rubberband_baseline, axis = 0, arr=self.X)

        else:
            # Initialise ray with the number of cores specified
            ray.init(num_cpus=self.n_jobs)

            # Get the baseline matrix from the ray jobs
            self.baseline = np.array(ray.get([self.rubberband_baseline_ray.remote(spectrum) 
            for spectrumm in np.apply_along_axis(lambda row: row, axis = 0, arr=self.X)]))

        return self