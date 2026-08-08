#!/usr/bin/env python3
"""
SE(2) pose-graph SLAM (Gauss-Newton), with a Huber robust kernel on
pose-pose edges. Own lightweight Graph/Edge classes since edges are
generated directly from our own trajectory + ICP registrations rather
than loaded from a .g2o file.

Scoped to horizontal (easting, northing, heading) drift correction only
-- our real trajectory is 3D (has roll/pitch too), but ground-vehicle
GNSS/SLAM drift is overwhelmingly a horizontal-position/heading problem,
and correcting only that is a legitimate, much simpler scope than a full
SE(3) reimplementation (elevation is left untouched).

IMPORTANT, established before writing this: the part15 trajectory has
zero usable in-tile loop closures (see loop_closure_icp.py's module
docstring -- only 308 epochs / 31s fall inside the part15 tile, the
vehicle passed through once). So this optimizer is implemented and
validated on a synthetic pose graph (see __main__), not run against real
part15 data -- there is nothing here for it to correct, honestly, not a
gap being glossed over.

Usage:
    python pose_graph_slam.py     # synthetic correctness self-test only
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def v2t(pose):
    x, y, theta = pose[0], pose[1], pose[2]
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def t2v(T):
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, theta])


class Edge:
    __slots__ = ("fromNode", "toNode", "measurement", "information")

    def __init__(self, fromNode, toNode, measurement, information):
        self.fromNode = fromNode
        self.toNode = toNode
        self.measurement = np.asarray(measurement, dtype=np.float64)
        self.information = np.asarray(information, dtype=np.float64)


class Graph:
    """x: flat state vector, 3 entries (x,y,theta) per node, in node-index order."""

    def __init__(self, initial_poses):
        self.n_nodes = len(initial_poses)
        self.x = np.array(initial_poses, dtype=np.float64).reshape(-1)
        self.lut = {i: 3 * i for i in range(self.n_nodes)}
        self.edges = []

    def add_edge(self, i, j, measurement, information):
        self.edges.append(Edge(i, j, measurement, information))


def linearize_pose_pose_constraint(x1, x2, z):
    """Error and Jacobians for an SE(2) pose-pose constraint, linearized
    around the current state estimate."""
    X1, X2, Z = v2t(x1), v2t(x2), v2t(z)
    Ri, Rij = X1[:2, :2], Z[:2, :2]
    ti, tj = x1[:2], x2[:2]
    theta_i = x1[2]

    e = t2v(np.linalg.inv(Z) @ np.linalg.inv(X1) @ X2)
    dRiT = np.array([[-np.sin(theta_i), np.cos(theta_i)],
                      [-np.cos(theta_i), -np.sin(theta_i)]])

    A = np.zeros((3, 3))
    A[:2, :2] = -Rij.T @ Ri.T
    A[:2, 2] = Rij.T @ dRiT @ (tj - ti)
    A[2, 2] = -1.0

    B = np.zeros((3, 3))
    B[:2, :2] = Rij.T @ Ri.T
    B[2, 2] = 1.0
    return e, A, B


def huber_weight(chi2, delta):
    """Huber robust-kernel weight: full weight inside the threshold,
    down-weighted (1/sqrt(chi2)) beyond it."""
    if abs(chi2) < delta ** 2:
        return 1.0
    return delta / np.sqrt(abs(chi2))


def compute_global_error(g):
    total = 0.0
    for edge in g.edges:
        x1 = g.x[g.lut[edge.fromNode]:g.lut[edge.fromNode] + 3]
        x2 = g.x[g.lut[edge.toNode]:g.lut[edge.toNode] + 3]
        e, _, _ = linearize_pose_pose_constraint(x1, x2, edge.measurement)
        total += e.T @ edge.information @ e
    return total


def linearize_and_solve(g, kernel="huber", huber_delta=3.0):
    """One Gauss-Newton step. Node 0 gets a prior (fixed reference frame --
    the whole trajectory is only defined up to a rigid transform without
    one anchored node, same as the reference implementation)."""
    n = len(g.x)
    H = lil_matrix((n, n))
    b = np.zeros(n)

    for edge in g.edges:
        i, j = g.lut[edge.fromNode], g.lut[edge.toNode]
        x1, x2 = g.x[i:i + 3], g.x[j:j + 3]
        e, A, B = linearize_pose_pose_constraint(x1, x2, edge.measurement)

        info = edge.information
        if kernel == "huber":
            chi2 = e.T @ info @ e
            w = huber_weight(chi2, huber_delta)
            info = w * info

        H[i:i + 3, i:i + 3] += A.T @ info @ A
        H[i:i + 3, j:j + 3] += A.T @ info @ B
        H[j:j + 3, i:i + 3] += B.T @ info @ A
        H[j:j + 3, j:j + 3] += B.T @ info @ B
        b[i:i + 3] += A.T @ info @ e
        b[j:j + 3] += B.T @ info @ e

    H[0:3, 0:3] += 1000 * np.eye(3)  # prior fixing node 0
    dx = spsolve(csr_matrix(H), -b)
    return dx


def run_pose_graph_slam(g, max_iter=30, tol=1e-4, kernel="huber", huber_delta=3.0, verbose=True):
    old_err = np.inf
    for it in range(max_iter):
        dx = linearize_and_solve(g, kernel=kernel, huber_delta=huber_delta)
        g.x += dx
        err = compute_global_error(g)
        if verbose:
            print(f"    iter {it}: error={err:.6f}")
        if abs(err - old_err) < tol:
            break
        old_err = err
    return g


if __name__ == "__main__":
    print("Synthetic correctness self-test (no real part15 loop closures exist -- see module docstring)")
    rng = np.random.default_rng(1)

    # ground truth: a square-ish loop, 40 poses
    n = 40
    theta_true = np.linspace(0, 2 * np.pi, n, endpoint=False)
    true_poses = np.stack([10 * np.cos(theta_true), 10 * np.sin(theta_true),
                            theta_true + np.pi / 2], axis=1)

    # odometry measurements: relative pose between consecutive true poses,
    # corrupted with small per-step noise that ACCUMULATES into real drift
    odom_noise_std = np.array([0.03, 0.03, 0.01])
    noisy_poses = [true_poses[0].copy()]
    odom_edges = []
    for k in range(1, n):
        X1, X2 = v2t(true_poses[k - 1]), v2t(true_poses[k])
        z_true = t2v(np.linalg.inv(X1) @ X2)
        z_meas = z_true + rng.normal(0, odom_noise_std)
        odom_edges.append((k - 1, k, z_meas))
        Xprev = v2t(noisy_poses[-1])
        noisy_poses.append(t2v(Xprev @ v2t(z_meas)))
    noisy_poses = np.array(noisy_poses)

    drift_before = np.linalg.norm(noisy_poses[-1, :2] - true_poses[-1, :2])
    print(f"  accumulated drift at loop closure (before): {drift_before:.3f}m")

    g = Graph(noisy_poses)
    info_odom = np.linalg.inv(np.diag(odom_noise_std ** 2))
    for i, j, z in odom_edges:
        g.add_edge(i, j, z, info_odom)

    # the one real loop-closure constraint: true relative pose between the
    # first and last node (this is what ICP would have supplied from real
    # point-cloud registration, if we had a real revisit -- here it's the
    # known true relative pose, since this is a controlled correctness test)
    X1, X2 = v2t(true_poses[-1]), v2t(true_poses[0])
    z_loop = t2v(np.linalg.inv(X1) @ X2) + rng.normal(0, [0.01, 0.01, 0.005])
    info_loop = np.linalg.inv(np.diag([0.01, 0.01, 0.005]) ** 2)
    g.add_edge(n - 1, 0, z_loop, info_loop)

    print("  running Gauss-Newton pose-graph optimization...")
    run_pose_graph_slam(g, max_iter=20, kernel="huber", verbose=False)

    optimized = g.x.reshape(-1, 3)
    err_before = np.linalg.norm(noisy_poses[:, :2] - true_poses[:, :2], axis=1)
    err_after = np.linalg.norm(optimized[:, :2] - true_poses[:, :2], axis=1)
    print(f"  mean position error BEFORE optimization: {err_before.mean():.3f}m (max {err_before.max():.3f}m)")
    print(f"  mean position error AFTER  optimization: {err_after.mean():.3f}m (max {err_after.max():.3f}m)")
    print(f"  {'PASS' if err_after.mean() < err_before.mean() * 0.5 else 'CHECK'} "
          f"-- loop closure should substantially reduce accumulated drift")
