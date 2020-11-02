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

    def __init__(self, n_jobs = None):

        self.n_jobs = n_jobs
   

    def transform(self, y):

        return self.y - self.baseline

    @ray.remote
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


    def fit(self, y):

        if self.n_jobs not None:

            # Initialise ray with the number of cores specified
            ray.init(num_cpus=self.n_jobs)
            self.y=y

            # Get the baseline matrix from the ray jobs
            self.baseline = np.array(ray.get([self.rubberband_baseline.remote(i) 
            for i in np.apply_along_axis(lambda row: row, axis = 0, arr=self.y)]))

            return self

        else:
            np.apply_along_axis(self.Kohler, axis = 0, arr=self.y)