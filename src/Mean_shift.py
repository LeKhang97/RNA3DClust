"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""

# Authors: Conrad Lee <conradlee@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Martino Sorbaro <martino.sorbaro@ed.ac.uk>

import warnings
from collections import defaultdict
from numbers import Integral, Real

import numpy as np
from sklearn._config import config_context
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted


@validate_params(
    {
        "X": ["array-like"],
        "quantile": [Interval(Real, 0, 1, closed="both")],
        "n_samples": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0, n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.

    This function takes time at least quadratic in `n_samples`. For large
    datasets, it is wise to subsample by setting `n_samples`. Alternatively,
    the parameter `bandwidth` can be set to a small value without estimating
    it.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input points.

    quantile : float, default=0.3
        Should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.

    n_samples : int, default=None
        The number of samples to use. If not given, all samples are used.

    random_state : int, RandomState instance, default=None
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    bandwidth : float
        The bandwidth parameter.
    """
    X = check_array(X)

    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
    nbrs.fit(X)

    bandwidth = 0.0
    for batch in gen_batches(len(X), 500):
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()

    return bandwidth / X.shape[0]


# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()["radius"]
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean
        my_mean = np.mean(points_within, axis=0)
        
        # If converged or at max_iter, adds the cluster
        if (
            np.linalg.norm(my_mean - my_old_mean) < stop_thresh
            or completed_iterations == max_iter
        ):
            break
        completed_iterations += 1
    #print(completed_iterations) #modified
    return tuple(my_mean), len(points_within), completed_iterations


@validate_params(
    {"X": ["array-like"]},
    prefer_skip_nested_validation=False,
)
def mean_shift(
    X,
    *,
    bandwidth=None,
    seeds=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    max_iter=300,
    n_jobs=None,
    quantile=0.3,
    kernel="flat",
    save_iterations=False,
):
    """Perform mean shift clustering of data using a flat kernel.

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input data.

    bandwidth : float, default=None
        Kernel bandwidth. If not None, must be in the range [0, +inf).

        If None, the bandwidth is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like of shape (n_seeds, n_features) or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    n_jobs : int, default=None
        The number of jobs to use for the computation. The following tasks benefit
        from the parallelization:

        - The search of nearest neighbors for bandwidth estimation and label
          assignments. See the details in the docstring of the
          ``NearestNeighbors`` class.
        - Hill-climbing optimization for all seeds.

        See :term:`Glossary <n_jobs>` for more details.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.17
           Parallel Execution using *n_jobs*.

    Returns
    -------

    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_mean_shift.py
    <sphx_glr_auto_examples_cluster_plot_mean_shift.py>`.
    """
    if kernel not in ['flat', 'gaussian']:
        raise ValueError("Invalid kernel. Choose 'flat' or 'gaussian'.")

    model = MeanShift(
        bandwidth=bandwidth,
        seeds=seeds,
        min_bin_freq=min_bin_freq,
        bin_seeding=bin_seeding,
        cluster_all=cluster_all,
        n_jobs=n_jobs,
        max_iter=max_iter,
        quantile=quantile,
        kernel=kernel,
        save_iterations=save_iterations,
    ).fit(X)
        
    return model.cluster_centers_, model.labels_

def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Find seeds for mean_shift.

    Finds seeds by first binning data onto a grid whose lines are
    spaced bin_size apart, and then choosing those bins with at least
    min_bin_freq points.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Input points, the same points that will be used in mean_shift.

    bin_size : float
        Controls the coarseness of the binning. Smaller values lead
        to more seeding (which is computationally more expensive). If you're
        not sure how to set this, set it to the value of the bandwidth used
        in clustering.mean_shift.

    min_bin_freq : int, default=1
        Only bins with at least min_bin_freq will be selected as seeds.
        Raising this value decreases the number of seeds found, which
        makes mean_shift computationally cheaper.

    Returns
    -------
    bin_seeds : array-like of shape (n_samples, n_features)
        Points used as initial kernel positions in clustering.mean_shift.
    """
    if bin_size == 0:
        return X

    # Bin points
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.round(point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    # Select only those bins as seeds which have enough members
    bin_seeds = np.array(
        [point for point, freq in bin_sizes.items() if freq >= min_bin_freq],
        dtype=np.float32,
    )
    if len(bin_seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds."
            % bin_size
        )
        return X
    bin_seeds = bin_seeds * bin_size
    return bin_seeds


class MeanShift(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        *,
        bandwidth=None,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        n_jobs=None,
        max_iter=300,
        quantile=0.3,
        kernel='flat',  # added parameter
        adaptive_bandwidth=False, # added parameter
        save_iterations=False # added parameter
    ):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.quantile = quantile
        self.kernel = kernel  # added parameter
        self.adaptive_bandwidth = adaptive_bandwidth  # added parameter
        self.save_iterations = save_iterations  # added parameter
        self.intermediate_means = []  # added parameter

    def fit(self, X, y=None):   
        X = self._validate_data(X)
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, quantile=self.quantile, n_jobs=self.n_jobs)
            #print(bandwidth)  # modified

        seeds = self.seeds
        if seeds is None:
            if self.bin_seeding:
                seeds = get_bin_seeds(X, bandwidth, self.min_bin_freq)
            else:
                seeds = X
        n_samples, n_features = X.shape
        center_intensity_dict = {}

        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)

        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(self._mean_shift_single_seed)(seed, X, nbrs, max_iter = self.max_iter)  # changed
            for seed in seeds
        )

        for i in range(len(seeds)):
            if all_res[i][1]:
                center_intensity_dict[all_res[i][0]] = all_res[i][1]


        self.n_iter_ = max([x[2] for x in all_res])
        if self.n_iter_ == self.max_iter:
            warnings.warn(
                "\nMeanShift algorithm may not converge. Try increasing the bandwidth.\n"
            )
        else:
            print(f"\nMeanShift algorithm converged after {self.n_iter_} iterations.\n")
        
        if self.save_iterations:
            self.intermediate_means = [x[3] for x in all_res]


        if not center_intensity_dict:
            raise ValueError(
                "No point was within bandwidth=%f of any seed. Try a different seeding"
                " strategy or increase the bandwidth."
                % bandwidth
            )

        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        unique = np.ones(len(sorted_centers), dtype=bool)
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=self.n_jobs).fit(
            sorted_centers
        )
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center], return_distance=False)[
                    0
                ]
                unique[neighbor_idxs] = 0
                unique[i] = 1
        cluster_centers = sorted_centers[unique]

        if self.adaptive_bandwidth:
            #print(self.adaptive_bandwidth)  # modified
            self.bandwidth = self._adaptive_bandwidth(X, cluster_centers)
            #print(self.bandwidth)  # modified

        if self.kernel == 'flat':
            self._assign_labels_flat(X, cluster_centers, bandwidth)
        elif self.kernel == 'gaussian':
            self._assign_labels_gaussian(X, cluster_centers, bandwidth)
        else:
            raise ValueError("Invalid kernel. Choose 'flat' or 'gaussian'.")

        #print(self.bandwidth, self.quantile)  # modified
        return self

    def _mean_shift_single_seed(self, my_mean, X, nbrs, max_iter):
        bandwidth = nbrs.get_params()["radius"]
        stop_thresh = 1e-3 * bandwidth
        completed_iterations = 0
        if self.save_iterations:
            intermediate_means = [my_mean]

        while True:
            i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
            points_within = X[i_nbrs]
            if len(points_within) == 0:
                break

            my_old_mean = my_mean

            if self.kernel == 'flat':
                my_mean = np.mean(points_within, axis=0)
            elif self.kernel == 'gaussian':
                weights = self.gaussian_kernel(
                    np.linalg.norm(points_within - my_mean, axis=1), bandwidth
                )
                my_mean = np.average(points_within, axis=0, weights=weights)
            
            if self.save_iterations:
                if type(self.save_iterations) == bool:
                    intermediate_means.append(my_mean)
                else:
                    if completed_iterations %(self.save_iterations[1] - self.save_iterations[0]) == 0:
                        intermediate_means.append(my_mean)
                    

            if (
                np.linalg.norm(my_mean - my_old_mean) < stop_thresh
                or completed_iterations == max_iter
            ):
                break

            # If adaptive bandwidth is enabled, update the bandwidth
            if self.adaptive_bandwidth:
                bandwidth = self._adaptive_bandwidth(X, my_mean.reshape(1, -1))
                #print(completed_iterations, bandwidth)  # modified
            
            completed_iterations += 1

        #print(completed_iterations, bandwidth)  # modified
        if self.save_iterations:
            return tuple(my_mean), len(points_within), completed_iterations, intermediate_means
        
        return tuple(my_mean), len(points_within), completed_iterations


    def gaussian_kernel(self, distance, bandwidth):  # changed
        return np.exp(-0.5 * (distance / bandwidth) ** 2)
    
    def _adaptive_bandwidth(self, X, cluster_centers):  # added method
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        distances, _ = nbrs.kneighbors(X)
        return np.median(distances)

    def _assign_labels_flat(self, X, cluster_centers, bandwidth):  # added method
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        labels = np.zeros(X.shape[0], dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels

    def _assign_labels_gaussian(self, X, cluster_centers, bandwidth):  # added method
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(cluster_centers)
        labels = np.zeros(X.shape[0], dtype=int)
        distances, idxs = nbrs.kneighbors(X)
        if self.cluster_all:
            labels = idxs.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= bandwidth
            labels[bool_selector] = idxs.flatten()[bool_selector]

        self.cluster_centers_, self.labels_ = cluster_centers, labels

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        with config_context(assume_finite=True):
            return pairwise_distances_argmin(X, self.cluster_centers_)
