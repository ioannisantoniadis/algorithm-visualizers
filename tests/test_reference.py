"""Each from-scratch algorithm checked against an independent reference: a library
implementation, a closed form, or finite differences. The visualizers are only worth
watching if the thing being animated is the real algorithm."""

import networkx as nx
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

import backprop.algorithm as BP
import dbscan.algorithm as DB
import dbscan.data as DD
import dijkstra.algorithm as DJ
import dijkstra.data as DJD
import gmmviz.algorithm as GM
import gmmviz.data as GD
import kalman.algorithm as KF
import kmeans.algorithm as KM
import kmeans.data as KD
import mcmc.algorithm as MC
import mst.algorithm as MA
import mst.data as MD
import pca.algorithm as PC
import pca.data as PD
import svm.algorithm as SV
import svm.data as SD
from fft.algorithm import fft_radix2, ifft_radix2

GENERATORS = ["make_blobs", "make_anisotropic", "make_varied"]


@pytest.mark.parametrize("n", [8, 64, 256, 1024])
def test_fft_matches_numpy_and_inverts(n):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    X, _ = fft_radix2(x, record=False)
    np.testing.assert_allclose(X, np.fft.fft(x), atol=1e-9)
    np.testing.assert_allclose(ifft_radix2(X), x, atol=1e-9)


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kmeans_matches_sklearn_from_same_init(gen, seed):
    X, _ = getattr(KD, gen)(seed=seed)
    init = X[np.random.default_rng(seed).choice(len(X), 4, replace=False)]
    final = KM.fit(X, 4, init_centroids=init.copy())[-1]
    ref = KMeans(4, init=init, n_init=1, max_iter=300, tol=0).fit(X)
    np.testing.assert_allclose(np.sort(final.centroids, 0), np.sort(ref.cluster_centers_, 0), atol=1e-6)


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("eps,min_samples", [(0.5, 5), (0.3, 4), (0.8, 8)])
def test_dbscan_matches_sklearn(gen, eps, min_samples):
    X, _ = getattr(DD, gen)(seed=1)
    labels = DB.fit(X, eps=eps, min_samples=min_samples)[-1].labels
    ref = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
    assert adjusted_rand_score(labels, ref) > 0.999
    np.testing.assert_array_equal(labels == -1, ref == -1)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("gen", GENERATORS)
def test_gmm_em_is_monotone_and_converges_to_an_em_fixed_point(gen):
    X, _ = getattr(GD, gen)(seed=2)
    snaps = GM.fit(X, 4, max_iter=200, random_state=0)
    lls = [s.log_likelihood for s in snaps]
    assert all(b >= a - 1e-8 for a, b in zip(lls, lls[1:])), "EM must never decrease the log-likelihood"
    s = snaps[-1]
    # One further EM step by an independent implementation barely moves the likelihood.
    ref = GaussianMixture(4, weights_init=s.weights, means_init=s.means,
                          precisions_init=np.linalg.inv(s.covs), max_iter=1, reg_covar=1e-6).fit(X)
    assert abs(ref.score(X) * len(X) - s.log_likelihood) / len(X) < 1e-3


def test_pca_matches_eigendecomposition_of_sample_covariance():
    X, _ = PD.make_anisotropic(seed=0)
    s = [x for x in PC.fit(X) if x.evals is not None][-1]
    evals, evecs = np.linalg.eigh(np.cov(X.T, ddof=1))
    np.testing.assert_allclose(s.evals, evals[::-1], rtol=1e-6)
    np.testing.assert_allclose(np.abs(np.sum(evecs[:, ::-1] * s.evecs, 0)), 1.0, atol=1e-8)


def _grid_graph(g):
    G = nx.DiGraph()
    rows, cols = g.shape
    for r in range(rows):
        for c in range(cols):
            if np.isfinite(g[r, c]):
                for nb in DJ._neighbors((r, c), rows, cols):
                    if np.isfinite(g[nb]):
                        G.add_edge((r, c), nb, weight=g[nb])
    return G


@pytest.mark.parametrize("gen", ["make_weighted_terrain", "make_maze", "make_random_obstacles"])
@pytest.mark.parametrize("algorithm", ["dijkstra", "astar"])
@pytest.mark.parametrize("heuristic", ["manhattan", "euclidean"])
def test_shortest_path_cost_is_optimal(gen, algorithm, heuristic):
    g = getattr(DJD, gen)(21, 21, seed=3)
    free = [tuple(p) for p in np.argwhere(np.isfinite(g))]
    start, goal = free[0], free[-1]
    G = _grid_graph(g)
    last = DJ.search(g, start, goal, algorithm=algorithm, heuristic_mode=heuristic)[-1]
    if nx.has_path(G, start, goal):
        assert last.phase == "found"
        assert last.dist[goal] == pytest.approx(nx.shortest_path_length(G, start, goal, weight="weight"))
    else:
        assert last.phase == "no_path"


@pytest.mark.parametrize("gen", ["make_random_euclidean", "make_clustered"])
@pytest.mark.parametrize("seed", range(4))
def test_kruskal_and_prim_find_the_minimum_spanning_tree(gen, seed):
    pos, edges = getattr(MD, gen)(seed=seed)
    G = nx.Graph()
    G.add_weighted_edges_from([(e.u, e.v, e.w) for e in edges])
    ref = nx.minimum_spanning_tree(G).size(weight="weight")
    assert MA.kruskal(pos, edges)[-1].total_weight == pytest.approx(ref)
    assert MA.prim(pos, edges)[-1].total_weight == pytest.approx(ref)


@pytest.mark.parametrize("gen,kernel", [("make_linear_separable", "linear"),
                                        ("make_linear_overlap", "linear"),
                                        ("make_moons", "rbf")])
def test_smo_svm_agrees_with_libsvm(gen, kernel):
    X, y = getattr(SD, gen)(seed=0)
    gamma = SV.default_gamma(X)
    s = SV.fit(X, y, kernel=kernel, C=1.0, gamma=gamma, max_epochs=200)[-1]
    ref = SVC(C=1.0, kernel=kernel, gamma=gamma).fit(X, y)
    assert s.converged
    # SMO stops at tol=1e-3 and libsvm at a tighter tolerance, so the two may disagree on
    # points sitting almost exactly on the decision boundary -- but on no point clearly
    # away from it.
    ref_decision = ref.decision_function(X)
    clear = np.abs(ref_decision) > 0.1
    assert clear.mean() > 0.8
    np.testing.assert_array_equal(np.sign(s.decision[clear]), np.sign(ref_decision[clear]))


def test_backprop_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(float)
    W, B = BP._init_params([2, 5, 3, 1], rng)
    _, acts = BP._forward(X, W, B)
    grad_w, _ = BP._backward(y, acts, W)

    def loss(Ws):
        return BP._bce(BP._forward(X, Ws, B)[1][-1], y)

    h = 1e-6
    for i in range(len(W)):
        for idx in np.ndindex(W[i].shape):
            plus = [w.copy() for w in W]
            minus = [w.copy() for w in W]
            plus[i][idx] += h
            minus[i][idx] -= h
            assert (loss(plus) - loss(minus)) / (2 * h) == pytest.approx(grad_w[i][idx], abs=1e-7)


def test_kalman_filter_matches_textbook_equations():
    rng = np.random.default_rng(0)
    n = 60
    true = np.cumsum(np.ones((n, 2)) * 0.5, 0)
    obs = true + rng.normal(scale=1.0, size=(n, 2))
    updates = [s for s in KF.fit(obs, true, dt=1.0, q=0.1, r=1.0, init_velocity_var=10.0) if s.phase == "update"]
    F, Q, H, R = KF._transition_matrix(1.0), KF._process_noise_cov(1.0, 0.1), KF._H, np.eye(2)
    x = np.array([obs[0, 0], obs[0, 1], 0.0, 0.0])
    P = np.diag([1.0, 1.0, 10.0, 10.0])
    for z, snap in zip(obs[1:], updates, strict=True):
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x, P = x + K @ (z - H @ x), (np.eye(4) - K @ H) @ P
        np.testing.assert_allclose(snap.mean, x, atol=1e-10)
        np.testing.assert_allclose(snap.cov, P, atol=1e-10)


def test_metropolis_hastings_recovers_the_gaussian_target():
    chain = MC.run_mcmc("gaussian", n_steps=60_000, proposal_sigma=1.5, seed=0)[-1].chain[5_000:]
    rho, sx, sy = 0.85, 1.8, 1.0  # mcmc/data.py _log_gaussian defaults
    target = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
    np.testing.assert_allclose(np.cov(chain.T), target, rtol=0.1, atol=0.05)
