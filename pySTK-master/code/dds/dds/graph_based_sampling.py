from itertools import permutations
import random
import time

import numpy as np
from numpy import linalg as LA
from scipy.sparse import csgraph
from sknetwork.topology.structure import get_largest_connected_component
import vtk

from .graph import Graph, ELLLikeGraph, graphless_neighbors


def sample_contrast_points(args, indices, tot_samples, vals_np, prob_vals, lcc_size=None):
    threshold = np.partition(vals_np, -tot_samples)[-tot_samples]
    low_degree_seeds = indices[vals_np < threshold]
    if lcc_size is not None:
        selected_low_degree_points = np.random.choice(low_degree_seeds, int(tot_samples - lcc_size), False)
    else:
        selected_low_degree_points = np.random.choice(low_degree_seeds, int(args.contrast_points * tot_samples), False)
    prob_vals[selected_low_degree_points] = 1
    return prob_vals
# End of contrast_points()


def gradient_lcc_sampling(args, dim, vals_np):
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    # print("3d shape: ", np.shape(vals_3d))
    vals_3d_grad = np.gradient(vals_3d)
    # print("gradient shape: ", np.shape(vals_3d_grad))
    vals_3d_grad = LA.norm(vals_3d_grad, axis=0)
    # print("normalized gradient shape: ", np.shape(vals_3d_grad))
    vals_3d_grad = vals_3d_grad.flatten()

    sparse = int(100 * (1.0 - args.percentage))
    graph = ELLLikeGraph.from_cube_sparse(dim, vals_3d_grad, sparse)
    
    csr = graph.to_csr()
    _, lcc_indices = get_largest_connected_component(csr, return_index=True)
    prob_vals[lcc_indices] = 1

    # Fill sample slots not taken up by LCC with random points
    if lcc_indices.size < tot_samples:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals,
                                           lcc_indices.size)

    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1

    return stencil
# End of gradient_lcc_sampling


def histogram_lcc_sampling(args, dim, vals_np):
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    hist_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    
    hist_bar_count, _ = np.histogram(vals_np, bins=args.nbins)
    # print("hist bar counts: ", hist_bar_count)
    frac = args.percentage
    tot_samples = frac * samples

    # create a dictionary first
    my_dict = dict()
    ind = 0
    for i in hist_bar_count:
        my_dict[ind] = i
        ind = ind + 1

    sorted_count = sorted(my_dict, key=lambda k: my_dict[k])

    # now distribute across bins
    target_bin_vals = int(tot_samples / args.nbins)
    new_count = np.copy(hist_bar_count)
    ind = 0
    remain_tot_samples = tot_samples
    for i in sorted_count:
        if my_dict[i] > target_bin_vals:
            val = target_bin_vals
        else:
            val = my_dict[i]
            # remain = target_bin_vals - my_dict[i]
        new_count[i] = val
        ind = ind + 1
        remain_tot_samples = remain_tot_samples - val
        if ind < args.nbins:
            target_bin_vals = int(remain_tot_samples / (args.nbins - ind))
    acceptance_hist = new_count / hist_bar_count

    bound_min = np.min(vals_np)
    bound_max = np.max(vals_np)

    # decide which bin this point goes to
    for i in range(samples):
        loc = vals_np[i]
        x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == args.nbins:
            x_id = x_id - 1
        hist_vals[i] = acceptance_hist[x_id]

    sparse = int(100 * (1.0 - args.percentage))
    graph = ELLLikeGraph.from_cube_sparse(dim, hist_vals, sparse)
    
    csr = graph.to_csr()
    _, lcc_indices = get_largest_connected_component(csr, return_index=True)
    prob_vals[lcc_indices] = 1

    # Fill sample slots not taken up by LCC with random points
    if lcc_indices.size < tot_samples:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals,
                                           lcc_indices.size)

    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    print("numsampled = ", stencil.sum())

    return stencil
# End of histogram_lcc_sampling


def max_sampling(args, vals_np):
    start = time.time()
    ## vals_np holds the RTData values
    ## x,y,z hold the locations of these points
    ## now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    # rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals)

    percent_to_sample = (1 - args.contrast_points) * args.percentage
    percentile = (1 - percent_to_sample) * 100
    threshold = np.percentile(vals_np, percentile)
    # print("percentile: ", percentile, " = ", threshold)
    # print("range: [", np.min(vals_np), ",", np.mean(vals_np), ",", np.median(vals_np), ",", np.max(vals_np), "]")
    prob_vals[vals_np >= threshold] = 1.0
    # print("nonzero prob_vals: ", np.count_nonzero(prob_vals))
    rand_vals = np.random.random_sample(samples)

    # stencil[argmax[prob_vals][:args.percentage] = 1
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage, "max_{0}".format(args.contrast_points))

    return stencil
# End of max_sampling


def lcc_rand_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples

    sparse = int(100 * (1.0 - (args.percentage * 2)))
    sparse = max(0, sparse)
    graph = ELLLikeGraph.from_cube_sparse(dim, vals_np, sparse)
    
    csr = graph.to_csr()
    _, lcc_indices = get_largest_connected_component(csr, return_index=True)
    if lcc_indices.size > frac:
        lcc_indices = np.random.choice(lcc_indices, min(lcc_indices.size, int(tot_samples)), replace=False)
    prob_vals[lcc_indices] = 1

    # Fill sample slots not taken up by LCC with random points
    if lcc_indices.size < tot_samples:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals,
                                           lcc_indices.size)

    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1

    return stencil
# End of lcc_rand_sampling


def lcc_sampling(args, dim, vals_np):
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = int(frac * samples)

    if tot_samples == 0:
        return stencil

    sparse = tot_samples
    graph = ELLLikeGraph.from_cube_sparse(dim, vals_np, sparse)
    if graph.num_edges > 0:
        csr = graph.to_csr()
        _, lcc_indices = get_largest_connected_component(csr, return_index=True)
    else:
        lcc_indices = np.asarray([], dtype=np.int32)
    prob_vals[lcc_indices] = 1

    # Fill sample slots not taken up by LCC with random points
    if lcc_indices.size < tot_samples:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals,
                                           lcc_indices.size)

    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1

    return stencil
# End of lcc_sampling


def lcc_skip_sampling(args, dim, vals_np):
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = int(frac * samples)

    if tot_samples == 0:
        return stencil

    sparse = tot_samples
    graph = ELLLikeGraph.from_cube_skip_sparse(dim, vals_np, sparse)
    if graph.num_edges > 0:
        csr = graph.to_csr()
        _, lcc_indices = get_largest_connected_component(csr, return_index=True)
    else:
        lcc_indices = np.asarray([], dtype=np.int32)
    prob_vals[lcc_indices] = 1

    # Fill sample slots not taken up by LCC with random points
    if lcc_indices.size < tot_samples:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals,
                                           lcc_indices.size)

    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1

    return stencil
# End of lcc_skip_sampling


def similarity_sampling(args, dim, vals_np):
    start = time.time()
    ## grad based sampling

    # compute gradient on the 3D data
    vals_3d = vals_np.reshape((dim[2], dim[1], dim[0]))
    # print(np.shape(vals_3d))

    vals_3d_grad = np.gradient(vals_3d)
    # print(np.shape(vals_3d_grad))
    vals_3d_sim = np.divide(1.0, vals_3d_grad)

    vals_3d_grad_mag = LA.norm(vals_3d_sim, axis=0)

    np.shape(vals_3d_grad_mag)

    grad_hist, edges = np.histogram(np.ndarray.flatten(vals_3d_grad_mag), bins=args.nbins)

    ## vals_np holds the RTData values
    ## x,y,z hold the locations of these points
    ## now apply sampling algorithms
    samples = np.size(vals_np)
    stencil_grad = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)


    # In[25]:

    ## create 1D acceptance histogram
    ## now start assiging points to 1D bins for each bin of 1D bin, start with largest gradients

    acceptance_hist_grad = np.zeros_like(grad_hist)
    acceptance_hist_grad_prob = np.zeros_like(grad_hist, dtype='float32')

    dist_counts = int(samples*args.percentage)
    remain_counts = dist_counts
    cur_count = 0
    for ii in range(args.nbins - 1, -1, -1):  ## looping from the most grad to least
        if remain_counts < grad_hist[ii]:
            cur_count = remain_counts
        else:
            cur_count = grad_hist[ii]
        acceptance_hist_grad[ii] = cur_count
        remain_counts = remain_counts - cur_count
        if grad_hist[ii] > 0.00000005:
            acceptance_hist_grad_prob[ii] = acceptance_hist_grad[ii] / grad_hist[ii]

    # print(np.shape(grad_hist), grad_hist, acceptance_hist_grad_prob)

    vals_3d_grad_mag_flattened = np.ndarray.flatten(vals_3d_grad_mag)
    bound_min = np.min(vals_3d_grad_mag_flattened)
    bound_max = np.max(vals_3d_grad_mag_flattened)

    # decide which bin this point goes to
    for i in range(samples):
        loc = vals_3d_grad_mag_flattened[i]
        x_id = int(args.nbins * (loc - bound_min) / (bound_max - bound_min))
        if x_id == args.nbins:
            x_id = x_id - 1
        prob_vals[i] = acceptance_hist_grad_prob[x_id]
        # generate a random number
        rand_vals[i] = random.uniform(0, 1)

    np.histogram(rand_vals, bins=args.nbins)
    np.size(np.where(rand_vals < prob_vals))

    stencil_grad = np.zeros_like(vals_np)
    stencil_grad[rand_vals < prob_vals] = 1

    # polydata = stencil_to_vtk(stencil_grad, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage, "similarity")

    return stencil_grad
# End of similarity_sampling()


def max_neighbor_sampling(args, dim, vals_np):
    start = time.time()
    ## vals_np holds the RTData values
    ## x,y,z hold the locations of these points
    ## now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')
    # selecting point + box around it
    tot_samples = tot_samples / 8

    initial_sampling_ratio = tot_samples / samples
    percentile = (1 - initial_sampling_ratio) * 100
    threshold = np.percentile(vals_np, percentile)
    # print("percentile: ", percentile, " = ", threshold)
    # print("range: [", np.min(vals_np), ",", np.mean(vals_np), ",", np.median(vals_np), ",", np.max(vals_np), "]")
    prob_vals[vals_np >= threshold] = 1.0
    # print("nonzero prob_vals: ", np.count_nonzero(prob_vals))
    rand_vals = np.random.random_sample(samples)

    selected_indices = np.nonzero(prob_vals)[0]
    selected_z = np.floor_divide(selected_indices, dim[0] * dim[1])
    selected_indices = selected_indices - (selected_z * dim[0] * dim[1])
    selected_y = np.floor_divide(selected_indices, dim[0])
    selected_x = np.mod(selected_indices, dim[0])

    # print("vals_np shape: ", vals_np.shape)
    # print("dim: ", dim)
    # print("prob_vals_nonzero shape: ", selected_indices.shape)

    for diff_x in [-1, 1]:  # range(-1, 2):
        for diff_y in [-1, 1]:  # range(-1, 2):
            for diff_z in [-1, 1]:  # range(-1, 2):
                x_ = np.clip(selected_x + diff_x, 0, dim[0] - 1)
                y_ = np.clip(selected_y + diff_y, 0, dim[1] - 1)
                z_ = np.clip(selected_z + diff_z, 0, dim[2] - 1)
                selected_1d = (z_ * dim[0] * dim[1]) + (y_ * dim[0]) + x_
                prob_vals[selected_1d] = 1

    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage, "max_neighbor")

    return stencil
# End of max_neighbor_sampling


def random_walk_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals)

    sampled = [False] * samples
    num_sampled = args.contrast_points * tot_samples

    # print("creating graph from cube")
    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    restart_chance = 0.1
    while num_sampled < tot_samples:
        while current_vertex < 0:
            current_vertex = random.randint(0, samples - 1)
            if graph.A[current_vertex, 0] == graph.num_vertices or graph.neighbors(current_vertex).size == 0:
                current_vertex = -1
        current_vertex = graph.random_neighbor(current_vertex)
        if not sampled[current_vertex]:
            num_sampled += 1
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage,
                # "random_walk_{0}_{1}r_{2}".format(args.sparse, restart_chance, args.contrast_points))

    return stencil
# End of random_walk_sampling


def value_weighted_random_walk_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = int(frac * samples)

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals)

    sampled = [False] * samples
    num_sampled = args.contrast_points * tot_samples

    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    current_vertex = -1
    s = time.time()
    starting_vertices = random.choices(np.arange(0, samples, dtype=np.int32), vals_np,
                                       k=min(vals_np.size, int(tot_samples * 5)))
    starter_idx = 0
    restart_chance = 0.1
    while num_sampled < tot_samples:
        while current_vertex < 0:
            # try:
            current_vertex = starting_vertices[starter_idx]
            # except IndexError as e:
            #     print("starter index: ", starter_idx, " num starting vertices: ", len(starting_vertices))
            #     raise e
            starter_idx = starter_idx + 1
            if graph.A[current_vertex, 0] == graph.num_vertices or graph.neighbors(current_vertex).size == 0:  # current_vertex not in graph.A or not graph.A[current_vertex]:
                current_vertex = -1
        neighbors = graph.neighbors(current_vertex)
        weights = vals_np[neighbors] ** 2
        current_vertex = random.choices(neighbors, weights, k=1)[0]
        if not sampled[current_vertex]:
            num_sampled += 1
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    e = time.time()
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1

    return stencil
# End of value_weighted_walk_sampling


def similarity_weighted_random_walk_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, np.arange(0, samples), tot_samples, vals_np, prob_vals)

    sampled = [False] * samples
    num_sampled = args.contrast_points * tot_samples

    # print("creating graph from cube")
    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    starting_vertices = random.choices(np.arange(0, samples, dtype=np.int32), vals_np, k=int(tot_samples))
    starter_idx = 0
    restart_chance = 0.1
    while num_sampled < tot_samples:
        while current_vertex < 0:
            current_vertex = starting_vertices[starter_idx]
            starter_idx = starter_idx + 1
            if graph.A[current_vertex, 0] == graph.num_vertices or graph.neighbors(current_vertex).size == 0:  # current_vertex not in graph.A or not graph.A[current_vertex]:
                current_vertex = -1
            # current_vertex = random.randint(0, samples - 1)
        # else:
        neighbors = graph.neighbors(current_vertex)
        weights = vals_np[neighbors] ** 2  # similarity
        distance = np.abs(weights - (vals_np[current_vertex] ** 2))
        distance = distance / (weights + (vals_np[current_vertex] ** 2))
        similarity = 1 - distance
        current_vertex = random.choices(neighbors, similarity, k=1)[0]
        if not sampled[current_vertex]:
            num_sampled += 1
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage,
                # "similarity_weighted_random_walk_{0}_{1}r_{2}".format(
                    # args.sparse, restart_chance, args.contrast_points))

    return stencil
# End of similarity_weighted_random_walk_sampling


def max_seeded_random_walk_sampling_old(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    percentile = (1 - args.percentage) * 100
    threshold = np.percentile(vals_np, percentile)
    # print("percentile: ", percentile, " = ", threshold)
    # print("range: [", np.min(vals_np), ",", np.mean(vals_np), ",", np.median(vals_np), ",", np.max(vals_np), "]")
    indices = np.arange(0, samples)
    seeds = indices[vals_np >= threshold]

    sampled = [False] * samples
    num_sampled = 0

    # print("creating graph from cube")
    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    seed_idx = 0
    restart_chance = 0.1
    while num_sampled < tot_samples:
        if current_vertex < 0:
            current_vertex = seeds[seed_idx]
            seed_idx = seed_idx + 1
        else:
            current_vertex = graph.random_neighbor(current_vertex)
        try:
            if not sampled[current_vertex]:
                num_sampled += 1
        except TypeError as e:
            print("Current vertex: ", current_vertex)
            raise e
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage, "max_seeded_random_walk_0.1_v2")

    return stencil
# End of max_seeded_random_walk_sampling_old


def max_seeded_random_walk_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    percentile = (1 - args.percentage) * 100
    # print("threshold percentile = ", percentile)
    threshold = np.percentile(vals_np, percentile)
    # print("percentile: ", percentile, " = ", threshold)
    # print("range: [", np.min(vals_np), ",", np.mean(vals_np), ",", np.median(vals_np), ",", np.max(vals_np), "]")
    indices = np.arange(0, samples)
    seeds = indices[vals_np >= threshold]
    np.random.shuffle(seeds)

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, indices, tot_samples, vals_np, prob_vals)

    sampled = [False] * samples
    num_sampled = args.contrast_points * tot_samples

    # print("creating graph from cube")
    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    seed_idx = 0
    restart_chance = 0.1
    while num_sampled < tot_samples:
        if current_vertex < 0:
            current_vertex = seeds[seed_idx]
            seed_idx = seed_idx + 1
        else:
            current_vertex = graph.random_neighbor(current_vertex)
        try:
            if not sampled[current_vertex]:
                num_sampled += 1
        except TypeError as e:
            print("Current vertex: ", current_vertex)
            raise e
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage,
                # "max_seeded_random_walk_{0}_{1}r_{2}".format(args.sparse, restart_chance, args.contrast_points))

    return stencil
# End of max_seeded_random_walk_sampling


def max_seeded_weighted_random_walk_sampling(args, dim, vals_np):
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    percentile = (1 - 2 * args.percentage) * 100
    threshold = np.percentile(vals_np, percentile)
    indices = np.arange(0, samples)
    seeds = indices[vals_np >= threshold]
    np.random.shuffle(seeds)

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, indices, tot_samples, vals_np, prob_vals)

    sampled = [False] * samples
    num_sampled = args.contrast_points * tot_samples

    # print("creating graph from cube")
    if args.sparse:
        graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    else:
        graph = ELLLikeGraph.from_cube(dim)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    seed_idx = 0
    restart_chance = 0.1
    while num_sampled < tot_samples:
        while current_vertex < 0:
            current_vertex = seeds[seed_idx]
            seed_idx = seed_idx + 1
            if graph.A[current_vertex, 0] == graph.num_vertices or graph.neighbors(current_vertex).size == 0:  # current_vertex not in graph.A or not graph.A[current_vertex]:
                current_vertex = -1
        # else:
        neighbors = graph.neighbors(current_vertex)
        weights = vals_np[neighbors]  # ** 2
        current_vertex = random.choices(neighbors, weights, k=1)[0]
            # current_vertex = graph.random_neighbor(current_vertex)
        if not sampled[current_vertex]:
            num_sampled += 1
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < restart_chance:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage,
                # "max_seeded_weighted_random_walk_{0}_{1}r_{2}".format(
                    # args.sparse, restart_chance, args.contrast_points))

    return stencil
# End of max_seeded_weighted_random_walk_sampling


def complex_max_seeded_random_walk_sampling(args, dim, vals_np):
    start = time.time()
    # vals_np holds the RTData values
    # x,y,z hold the locations of these points
    # now apply sampling algorithms
    samples = np.size(vals_np)
    stencil = np.zeros_like(vals_np)
    prob_vals = np.zeros_like(vals_np)
    rand_vals = np.zeros_like(vals_np)
    frac = args.percentage
    tot_samples = frac * samples
    # print('looking for', tot_samples, 'samples')

    percentile = (1 - args.percentage) * 100
    threshold = np.percentile(vals_np, percentile)
    indices = np.arange(0, samples)
    seeds = indices[vals_np >= threshold]
    np.random.shuffle(seeds)

    if args.contrast_points > 0.0:
        prob_vals = sample_contrast_points(args, indices, tot_samples, vals_np, prob_vals)
        # threshold2 = np.percentile(vals_np, 25)
        # low_degree_seeds = indices[vals_np < threshold2]
        # selected_low_degree_points = np.random.choice(low_degree_seeds, int(0.01 * tot_samples), False)
        # prob_vals[selected_low_degree_points] = 1

    sampled = [False] * samples
    num_sampled = int(args.contrast_points * tot_samples)

    # print("creating graph from cube")
    graph = ELLLikeGraph.from_cube_sparse(dim, vals_np)
    # graph = Graph.from_cube_sparse(dim[0], dim[1], dim[2], vals_np)
    # print("done creating graph from cube")
    current_vertex = -1
    # print("sampling")
    seed_idx = 0
    while num_sampled < tot_samples:
        if current_vertex < 0:
            current_vertex = seeds[seed_idx]
            seed_idx = seed_idx + 1
        else:
            current_vertex = graph.random_neighbor(current_vertex)
        if not sampled[current_vertex]:
            num_sampled += 1
        sampled[current_vertex] = True
        prob_vals[current_vertex] = 1
        if random.random() < 0.1:
            current_vertex = -1
    # print("done sampling")
    rand_vals = np.random.random_sample(samples)
    stencil[rand_vals < prob_vals] = 1
    # print("nonzero stencil vals: ", np.count_nonzero(stencil))
    # print("Collecting samples:", np.sum(stencil))

    # polydata = stencil_to_vtk(stencil, name, x, y, z, vals_np, j_list, args.fill)
    # save_sample(polydata, filename, args.percentage, "complex_max_seeded_random_walk_0.01_{0}".format(args.contrast_points))

    return stencil
# End of complex_max_seeded_random_walk_sampling
