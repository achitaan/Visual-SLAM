"""Portable SE(3) pose graph with a fixed first camera-to-world pose."""
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from kitti import validate_pose


def graph_vertex_groups(first_indices, second_indices, count):
    """Color free vertices so no edge depends on two vertices in one group."""
    neighbours = {vertex: set() for vertex in range(1, count)}
    for first, second in zip(first_indices, second_indices):
        if first and second:
            neighbours[first].add(second)
            neighbours[second].add(first)
    colors = {}
    for vertex in sorted(neighbours, key=lambda v: (-len(neighbours[v]), v)):
        used = {colors[other] for other in neighbours[vertex] if other in colors}
        colors[vertex] = next(color for color in range(count) if color not in used)
    groups = [[vertex for vertex in sorted(colors) if colors[vertex] == color] for color in range(max(colors.values()) + 1)]
    rows = {vertex: (6 * np.flatnonzero((first_indices == vertex) | (second_indices == vertex))[:, None] + np.arange(6)).ravel() for vertex in neighbours}
    return groups, rows


def dense_graph_jacobian(residual, parameters, groups, rows):
    """Grouped forward differences, expanded to a dense Jacobian for SVD."""
    baseline = residual(parameters)
    jacobian = np.zeros((len(baseline), len(parameters)))
    step = np.sqrt(np.finfo(float).eps) * np.where(parameters >= 0, 1., -1.) * np.maximum(1., np.abs(parameters))
    for group in groups:
        for coordinate in range(6):
            columns = np.array([(vertex - 1) * 6 + coordinate for vertex in group])
            perturbed = parameters.copy()
            perturbed[columns] += step[columns]
            difference = residual(perturbed) - baseline
            for vertex, column in zip(group, columns):
                jacobian[rows[vertex], column] = difference[rows[vertex]] / (perturbed[column] - parameters[column])
    return jacobian


def optimize(poses, edges, max_evaluations=100, diagnostics=None):
    if not poses:
        return []
    for pose in poses:
        validate_pose(pose)
    if len(poses) == 1 or not edges:
        return [pose.copy() for pose in poses]
    weights = []
    for first, second, measurement, information, _ in edges:
        if not (0 <= first < len(poses) and 0 <= second < len(poses)) or first == second:
            raise ValueError("Pose graph edge references an invalid vertex")
        validate_pose(measurement)
        info = np.asarray(information)
        if info.shape != (6, 6) or not np.isfinite(info).all() or not np.allclose(info, info.T):
            raise ValueError("Information matrix must be finite symmetric 6x6")
        # W.T W = information; Cholesky returns the opposite orientation.
        weights.append(np.linalg.cholesky(info).T)
    initial = np.array([np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]] for pose in poses[1:]])
    first_indices = np.array([edge[0] for edge in edges])
    second_indices = np.array([edge[1] for edge in edges])
    measurements = np.array([edge[2] for edge in edges])
    measurement_inverse_rotation = measurements[:, :3, :3].transpose(0, 2, 1)
    weights = np.array(weights)

    def unpack(parameters):
        values = parameters.reshape(-1, 6)
        result = np.repeat(np.eye(4)[None], len(poses), axis=0)
        result[0] = poses[0]
        result[1:, :3, :3] = Rotation.from_rotvec(values[:, :3]).as_matrix()
        result[1:, :3, 3] = values[:, 3:]
        return result

    def residual(parameters):
        current = unpack(parameters)
        first, second = current[first_indices], current[second_indices]
        inverse_rotation = first[:, :3, :3].transpose(0, 2, 1)
        # Rigid inverses avoid general matrix inversion for every edge/evaluation.
        relative_rotation = inverse_rotation @ second[:, :3, :3]
        relative_translation = np.einsum('nij,nj->ni', inverse_rotation, second[:, :3, 3] - first[:, :3, 3])
        delta_rotation = measurement_inverse_rotation @ relative_rotation
        delta_translation = np.einsum('nij,nj->ni', measurement_inverse_rotation, relative_translation - measurements[:, :3, 3])
        errors = np.c_[Rotation.from_matrix(delta_rotation).as_rotvec(), delta_translation]
        return np.einsum('nij,nj->ni', weights, errors).ravel()

    sparsity = lil_matrix((6 * len(edges), 6 * (len(poses) - 1)), dtype=int)
    for index, (first, second, *_rest) in enumerate(edges):
        for vertex in (first, second):
            if vertex > 0:
                sparsity[index * 6:(index + 1) * 6, (vertex - 1) * 6:vertex * 6] = 1
    options = dict(max_nfev=max_evaluations, loss="huber", f_scale=1.0,
                   x_scale=np.tile([.1, .1, .1, 1., 1., 1.], len(poses) - 1))
    # Small anchored graphs are ill-conditioned: approximate LSMR steps can
    # stall for thousands of iterations. SVD solves the same objective reliably.
    # Bound the dense Jacobian; larger maps retain the sparse solve.
    dense_bytes = 6 * len(edges) * initial.size * np.dtype(float).itemsize
    started = time.perf_counter()
    def robust_cost(values):
        absolute = np.abs(values)
        return float(np.where(absolute <= 1., .5 * values**2, absolute - .5).sum())
    initial_cost = robust_cost(residual(initial.ravel()))
    if len(poses) <= 300 and dense_bytes <= 64 * 1024 * 1024:
        groups, rows = graph_vertex_groups(first_indices, second_indices, len(poses))
        jacobian_calls = 0
        def jacobian(parameters):
            nonlocal jacobian_calls
            jacobian_calls += 1
            if diagnostics is not None and jacobian_calls % 25 == 0:
                print(f'Graph solve: {jacobian_calls} linearizations, cost {robust_cost(residual(parameters)):.8g}, elapsed {time.perf_counter() - started:.1f}s', flush=True)
            return dense_graph_jacobian(residual, parameters, groups, rows)
        options['jac'] = jacobian
        options['tr_solver'] = 'exact'
        solver = 'dense SVD with grouped forward differences'
    else:
        options.update(jac_sparsity=sparsity.tocsr(), tr_solver='lsmr',
                       tr_options={"atol": 1e-10, "btol": 1e-10,
                                   "maxiter": max(1000, 10 * initial.size)})
        solver = 'sparse LSMR'
    result = least_squares(residual, initial.ravel(), **options)
    if diagnostics is not None:
        diagnostics.update(solver=solver, success=bool(result.success), function_evaluations=result.nfev,
                           jacobian_evaluations=result.njev, initial_cost=initial_cost, final_cost=float(result.cost),
                           optimality=float(result.optimality), message=result.message,
                           elapsed_s=time.perf_counter() - started, dense_jacobian_bytes=dense_bytes)
    if not np.isfinite(result.x).all() or not result.success:
        raise RuntimeError(f"Pose graph optimization did not converge: {result.message}; "
                           f"evaluations={result.nfev}, cost={result.cost:.8g}, optimality={result.optimality:.8g}")
    return list(unpack(result.x))
