import math
import matplotlib.pyplot as plt
import random

import numpy as np
import pymp
from scipy.sparse import csr_matrix
import seaborn as sns


class Graph:
    """A stripped-down directed graph representation."""
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.num_edges = 0
        self.A = dict()
        # print("Creating self.A")
        # for i in range(num_vertices):
        #     self.A.append(list())  # pymp.shared.list())
        # print("Done creating self.A")
    # End of __init__()

    def add_edge(self, origin, destination):
        if origin < 0 or destination < 0:
            return
        if origin >= self.num_vertices or destination >= self.num_vertices:
            return
        self.A[origin].append(destination)
        self.num_edges = self.num_edges + 1
    # End of add_edge()

    def neighbors(self, vertex):
        return self.A[vertex]
    # End of neighbors()

    def random_neighbor(self, vertex):
        return random.choice(self.A[vertex])
    # End of random_neighbor()

    @classmethod
    def from_cube(cls, dim_x, dim_y, dim_z):
        num_vertices = dim_x * dim_y * dim_z
        graph = Graph(num_vertices)
        idx = np.arange(0, num_vertices)
        z = pymp.shared.array(np.size(idx), dtype=np.float64)
        z[:] = np.floor_divide(idx, dim_x * dim_y)
        idx = idx - (z * dim_x * dim_y)
        y = pymp.shared.array(np.size(idx), dtype=np.float64)
        y[:] = np.floor_divide(idx, dim_x)
        x = pymp.shared.array(np.size(idx), dtype=np.float64)
        x[:] = np.mod(idx, dim_x)
        ## parallel example
        dictionaries = pymp.shared.list()
        edges = pymp.shared.list()
        ## parallel example
        with pymp.Parallel(8) as p:
            my_dict = dict()
            my_num_edges = 0
            for vertex in p.range(num_vertices):
                my_dict[vertex] = list()
                for diff_x in range(-1, 2):
                    x_ = x[vertex] + diff_x
                    if x_ < 0 or x_ >= dim_x: continue
                    for diff_y in range(-1, 2):
                        y_ = y[vertex] + diff_y
                        if y_ < 0 or y_ >= dim_y: continue
                        for diff_z in range(-1, 2):
                            z_ = z[vertex] + diff_z
                            if z_ < 0 or z_ >= dim_z: continue
                            destination = int((z_ * dim_x * dim_y) + (y_ * dim_x) + x_)
                            # self.A[vertex].append(destination)
                            # graph.num_edges += 1
                            my_dict[vertex].append(destination)
                            my_num_edges += 1
                            # graph.add_edge(vertex, destination)
            dictionaries.append(my_dict)
            edges.append(my_num_edges)
        [graph.A.update(d) for d in dictionaries]
        graph.num_edges = np.sum(edges)
        print("Generated graph with {0} edges / {1}".format(graph.num_edges, graph.num_vertices ** 2))
        return graph

    @classmethod
    def from_cube_sparse(cls, dim_x, dim_y, dim_z, vals_np):
        num_vertices = dim_x * dim_y * dim_z
        graph = Graph(num_vertices)
        idx = np.arange(0, num_vertices)
        z = pymp.shared.array(np.size(idx), dtype=np.float64)
        z[:] = np.floor_divide(idx, dim_x * dim_y)
        idx = idx - (z * dim_x * dim_y)
        y = pymp.shared.array(np.size(idx), dtype=np.float64)
        y[:] = np.floor_divide(idx, dim_x)
        x = pymp.shared.array(np.size(idx), dtype=np.float64)
        x[:] = np.mod(idx, dim_x)
        percentile_25 = np.percentile(vals_np, 25)
        dictionaries = pymp.shared.list()
        edges = pymp.shared.list()
        ## parallel example
        with pymp.Parallel(8) as p:
            my_dict = dict()
            my_num_edges = 0
            for vertex in p.range(num_vertices):
                my_dict[vertex] = list()
                weight = vals_np[vertex]
                if weight < percentile_25: continue
                for diff_x in range(-1, 2):
                    x_ = x[vertex] + diff_x
                    if x_ < 0 or x_ >= dim_x: continue
                    for diff_y in range(-1, 2):
                        y_ = y[vertex] + diff_y
                        if y_ < 0 or y_ >= dim_y: continue
                        for diff_z in range(-1, 2):
                            z_ = z[vertex] + diff_z
                            if z_ < 0 or z_ >= dim_z: continue
                            destination = int((z_ * dim_x * dim_y) + (y_ * dim_x) + x_)
                            destination_weight = vals_np[destination]
                            if destination_weight < percentile_25: continue
                            # if vertex < destination:
                            similarity = 1.0 - abs(destination_weight - weight) / (weight + destination_weight)
                            if similarity > 0.975:
                                my_dict[vertex].append(destination)
                                my_num_edges += 1
                                # graph.add_edge(vertex, destination)
            dictionaries.append(my_dict)
            edges.append(my_num_edges)
        [graph.A.update(d) for d in dictionaries]
        graph.num_edges = np.sum(edges)
        # sns.set_theme()
        # axes = sns.histplot(data=similarities, kde=True)
        # plt.savefig("value_similarities.png")
        # exit(-100)
        print("Generated graph with {0} edges / {1}".format(graph.num_edges, graph.num_vertices ** 2))
        degrees = np.zeros_like(vals_np)
        for vertex in range(graph.num_vertices):
            if vertex in graph.A:
                degrees[vertex] = len(graph.A[vertex])
        return graph
    # End of from_cube_sparse()


class ELLLikeGraph:
    """A stripped-down directed graph representation."""
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.num_edges = 0
        self.A = np.zeros((num_vertices, 27), dtype=np.int32)
        # print("Creating self.A")
        # for i in range(num_vertices):
        #     self.A.append(list())  # pymp.shared.list())
        # print("Done creating self.A")
    # End of __init__()

    def add_edge(self, origin, destination):
        if origin < 0 or destination < 0:
            return
        if origin >= self.num_vertices or destination >= self.num_vertices:
            return
        self.A[origin].append(destination)
        self.num_edges = self.num_edges + 1
    # End of add_edge()

    def neighbors(self, vertex):
        current_neighbors = self.A[vertex][1:]  # [current_vertex][0] == current_vertex (self-edge)
        return current_neighbors[current_neighbors < self.num_vertices]
    # End of neighbors()

    def random_neighbor(self, vertex):
        current_neighbors = self.A[vertex][1:]  # [current_vertex][0] == current_vertex (self-edge)
        current_neighbors = current_neighbors[current_neighbors < self.num_vertices]
        return random.choice(current_neighbors)
    # End of random_neighbor()

    def to_csr(self):
        data = np.ones(self.num_edges, dtype=np.int32)
        # indices = np.zeros(self.num_edges, dtype=np.int32)
        indices = self.A[:, 1:].flatten()
        indices = indices[indices < self.num_vertices]
        index_ptr = np.zeros(self.num_vertices + 1, dtype=np.int32)
        index_ptr[1:] = (self.A[:, 1:] < self.num_vertices).sum(axis=1).cumsum()
        return csr_matrix((data, indices, index_ptr), shape=(self.num_vertices, self.num_vertices))
    # End of to_csr()

    @classmethod
    def from_cube(cls, dim):
        dim_x = dim[0]
        dim_y = dim[1]
        dim_z = dim[2]
        num_vertices = dim_x * dim_y * dim_z
        graph = ELLLikeGraph(num_vertices)
        # convert vals_np to x_idx and y_idx
        idx = np.arange(0, num_vertices)
        z_idx = np.floor_divide(idx, dim[0] * dim[1], dtype=np.float64)
        idx = idx - (z_idx * dim[0] * dim[1])
        y_idx = np.floor_divide(idx, dim[0], dtype=np.float64)
        x_idx = np.mod(idx, dim[0], dtype=np.float64)
        for diff_x in [0, 1, -1]:
            x_ = x_idx + diff_x
            x_[(x_ == -1) | (x_ == dim[0])] = np.NAN
            for diff_y in [0, 1, -1]:
                y_ = y_idx + diff_y
                y_[(y_ == -1) | (y_ == dim[1])] = np.NAN
                for diff_z in [0, 1, -1]:
                    z_ = z_idx + diff_z
                    z_[(z_ == -1) | (z_ == dim[2])] = np.NAN
                    indices = (z_ * dim[0] * dim[1]) + (y_ * dim[0]) + x_
                    indices = np.nan_to_num(indices, nan=num_vertices).astype(np.int32)
                    neighbors_idx = (diff_z * 3 * 3) + (diff_y * 3) + diff_x
                    graph.A[:, neighbors_idx] = indices
        graph.num_edges = np.array(graph.A[:, 1:] < num_vertices).sum()
        print("Generated graph with {0} edges / {1}".format(graph.num_edges, graph.num_vertices ** 2))
        return graph

    @classmethod
    def from_cube_sparse(cls, dim, vals_np, threshold_index=100):
        """Create a sparse graph with edges only between the top `threshold_index` vertices when sorted using `vals_np`.
        """
        # print("sparse = ", sparse)
        dim_x = dim[0]
        dim_y = dim[1]
        dim_z = dim[2]
        num_vertices = dim_x * dim_y * dim_z
        graph = ELLLikeGraph(num_vertices)
        # convert vals_np to x_idx and y_idx
        idx = np.arange(0, num_vertices)
        # if the dtype is smaller than float64, the division doesn't always work out properly. E.g. for 512 cube, max(z_idx) ends up being 512 instead of 511
        z_idx = np.floor_divide(idx, dim[0] * dim[1], dtype=np.float64)
        idx = idx - (z_idx * dim[0] * dim[1])
        y_idx = np.floor_divide(idx, dim[0], dtype=np.float64)
        x_idx = np.mod(idx, dim[0], dtype=np.float64)
        threshold = np.partition(vals_np, -threshold_index)[-threshold_index]
        values = np.append(vals_np, -1)
        for diff_x in [0, 1, -1]:
            x_ = x_idx + diff_x
            x_[(x_ == -1) | (x_ == dim[0])] = np.NAN
            for diff_y in [0, 1, -1]:
                y_ = y_idx + diff_y
                y_[(y_ == -1) | (y_ == dim[1])] = np.NAN
                for diff_z in [0, 1, -1]:
                    z_ = z_idx + diff_z
                    z_[(z_ == -1) | (z_ == dim[2])] = np.NAN
                    indices = (z_ * dim[0] * dim[1]) + (y_ * dim[0]) + x_
                    indices = np.nan_to_num(indices, nan=num_vertices).astype(np.int32)
                    indices[values[indices] < threshold] = num_vertices  # no edges to low-value points
                    neighbors_idx = (diff_z * 3 * 3) + (diff_y * 3) + diff_x
                    graph.A[:, neighbors_idx] = indices
        graph.A[vals_np < threshold, :] = num_vertices  # no edges from low-value points
        graph.num_edges = np.array(graph.A[:, 1:] < num_vertices).sum()
        return graph
    # End of from_cube_sparse()


def graphless_neighbors(vals_np, dim):
    neighbors = np.zeros((vals_np.size, 27), dtype=np.int32)
    # convert vals_np to x_idx and y_idx
    idx = np.arange(0, np.size(vals_np))
    z_idx = np.floor_divide(idx, dim[0] * dim[1], dtype=np.float32)
    idx = idx - (z_idx * dim[0] * dim[1])
    y_idx = np.floor_divide(idx, dim[0], dtype=np.float32)
    x_idx = np.mod(idx, dim[0], dtype=np.float32)
    for diff_x in [0, 1, -1]:
        x_ = x_idx + diff_x
        x_[(x_ == -1) | (x_ == dim[0])] = np.NAN
        for diff_y in [0, 1, -1]:
            y_ = y_idx + diff_y
            y_[(y_ == -1) | (y_ == dim[1])] = np.NAN
            for diff_z in [0, 1, -1]:
                z_ = z_idx + diff_z
                z_[(z_ == -1) | (z_ == dim[2])] = np.NAN
                indices = (z_ * dim[0] * dim[1]) + (y_ * dim[0]) + x_
                indices = np.nan_to_num(indices, nan=np.size(vals_np)).astype(np.int32)
                neighbors_idx = (diff_z * 3 * 3) + (diff_y * 3) + diff_x
                neighbors[:, neighbors_idx] = indices
    return neighbors
# End of graphless_neighbors()
