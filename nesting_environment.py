import copy
import os
import random
import random
import time

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from PIL import ImageDraw
from PIL.Image import Image
from rasterio._features import MergeAlg
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
import rasterio.features
import math
random.seed(0)
np.random.seed(0)
# scipy.rand.s



class Graph:
    def __init__(self, num_nodes):
        self.g = {k: [] for k in range(num_nodes)}

    def add_edge(self, i, j):
        if i == j:
            return
        if j not in self.g[i]:
            self.g[i].append(j)

        if i not in self.g[j]:
            self.g[j].append(i)

    def add_node(self, n):
        self.g[n] = []

    def get_all_nodes(self):
        return list(self.g.keys())

    def remove_node(self, n):
        self.g.pop(n)
        for k in self.g.keys():
            if n in self.g[k]:
                self.g[k].remove(n)

    def get_neighbours(self, n):
        return self.g[n]

    def merge_nodes(self, i, j):
        new_node = max(self.g.keys()) + 1
        self.add_node(new_node)
        for n in self.get_neighbours(i):
            self.add_edge(new_node, n)

        for n in self.get_neighbours(j):
            self.add_edge(new_node, n)

        self.remove_node(i)
        self.remove_node(j)

        return new_node


class NestEnvConfig:
    def __init__(self):
        # self.min_pts = 150
        # self.max_pts = 300

        self.min_pts = 30
        self.max_pts = 50

        self.border_min_pts = 0
        self.border_max_pts = 0

        self.min_num_of_polygons_after_merge = 30
        self.max_num_of_polygons_after_merge = 120

        self.poly_dropping_target_efficiency = 0.85

        self.poly_shrink_buffer_range = (0.0001, 0.001)

        self.raster_width=1000
        self.raster_height=1000

        self.show_state = True


class NestingEnvironment:
    def __init__(self, cfg:NestEnvConfig):
        self.cfg = cfg
        self.state = None
        self.current_polys = None


    def reset(self):

        self.build_state()

        self.initial_polys = copy.deepcopy(self.current_polys)

        # self.generate_overlap()

        return self.state

    def get_border_padding_pts(self):
        SIZE = 1.0
        points = []
        points.append((-SIZE * 3, -SIZE * 3))
        points.append((-SIZE * 3, SIZE * 4))
        points.append((SIZE * 4, -SIZE * 3))
        points.append((SIZE * 4, SIZE * 4))
        return np.array(points)

    def get_bordering_pts(self):
        num_pts = random.randint(self.cfg.border_min_pts, self.cfg.border_max_pts)
        pts_delta = np.random.random((num_pts,1))
        border_segments = np.array([
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]]
        ])

        selected_border = np.random.choice(4, size=num_pts)
        selected_border_segments = border_segments[selected_border]
        selected_border_pts = selected_border_segments[:,0,:] + \
                              (selected_border_segments[:,1,:] - selected_border_segments[:,0,:]) * pts_delta
        return selected_border_pts

    def generate_overlap(self):
        pass

    def step(self, actions):
        bounds = [rasterio.features.bounds(poly) for poly in self.current_polys]
        # top_right_bounds = [b[2:][::-1] for b in bounds]
        # lower_left_bounds = [b[:2][::-1] for b in bounds]

        normalized_poly_actions = []

        for poly_action, (left, bottom, right, top) in zip(actions, bounds):
            poly_delta_x, poly_delta_y = poly_action

            if right + poly_delta_x >= cfg.raster_width:
                poly_delta_x = cfg.raster_width - right
            if left + poly_delta_x < 0:
                poly_delta_x = -left

            if top + poly_delta_y >= cfg.raster_height:
                poly_delta_y = cfg.raster_height - top
            if bottom + poly_delta_y < 0:
                poly_delta_y = -bottom

            normalized_poly_actions.append([poly_delta_x, poly_delta_y])

        self.current_polys = [affinity.translate(poly, b[0], b[1]) for poly, b in
                       zip(self.current_polys, normalized_poly_actions)]

        self.draw_state()

    def draw_state(self):

        bounds = [rasterio.features.bounds(poly) for poly in self.current_polys]
        lower_left_bounds = [b[:2] for b in bounds]
        top_right_bounds = [b[2:] for b in bounds]
        integer_lower_left_bounds = [[int(b[0]), int(b[1])] for b in lower_left_bounds]
        integer_top_right_bounds = [ [int(math.ceil(b[0])), int(math.ceil(b[1]))] for b in top_right_bounds]
        moved_top_right_bounds = [[tr[0] - ll[0], tr[1] - ll[1]] for tr,ll in zip(integer_top_right_bounds, integer_lower_left_bounds)]

        moved_polys = [affinity.translate(poly, -b[0], -b[1]) for poly, b in zip(self.current_polys, integer_lower_left_bounds)]

        moved_top_right_bounds = [l[::-1] for l in moved_top_right_bounds]
        integer_lower_left_bounds = [l[::-1] for l in integer_lower_left_bounds]

        poly_rasters = [rasterio.features.rasterize([poly], out_shape=trb) for poly, trb in zip(moved_polys, moved_top_right_bounds)]

        state = np.zeros((self.cfg.raster_width, self.cfg.raster_height))
        for idx, (raster, coords) in enumerate(zip(poly_rasters, integer_lower_left_bounds)):
            raster_shape = raster.shape
            state[coords[0]: coords[0] + raster_shape[0], coords[1]: coords[1] + raster_shape[1]] =\
                state[coords[0]: coords[0] + raster_shape[0], coords[1]: coords[1] + raster_shape[1]] + raster * (idx + 1)

        self.state = state

        if self.cfg.show_state:
            plt.imshow(self.state)
            plt.show()


    def build_state(self):
        num_pts = random.randint(self.cfg.min_pts, self.cfg.max_pts)
        pts = np.random.random((num_pts, 2))
        border_pts = self.get_bordering_pts()
        border_pad_pts = self.get_border_padding_pts()
        all_pts = np.concatenate([pts, border_pts, border_pad_pts], axis=0)

        vor = Voronoi(all_pts, incremental=True)

        start = time.time()

        all_initial_neighbours = vor.point_region[vor.ridge_points]
        kept_region_idxs = []
        new_regions = []
        old_to_new_idx = {}
        for initial_idx, r in enumerate(vor.regions):
            if all(list(map(lambda v: v != -1, r))):
                    new_regions.append(r)
                    kept_region_idxs.append(initial_idx)
                    old_to_new_idx[initial_idx] = len(new_regions) - 1

        kept_region_idxs = set(kept_region_idxs)
        all_neighbours = []
        for neighbours in all_initial_neighbours:
            if neighbours[0] in kept_region_idxs and neighbours[1] in kept_region_idxs:
                all_neighbours.append([old_to_new_idx[neighbours[0]], old_to_new_idx[neighbours[1]]])

        all_neighbours = np.array(all_neighbours)
        end = time.time()
        print("Time consumed in computing neighbours: ", end - start)
        vor.regions = new_regions
        final_regions = []

        border_clipping_regions_polys = [
            [(-10, -10),(10, -10), (10, 0), (-10, 0)],
            [(1,-10), (10, -10), (10, 10), (1, 10)],
            [(-10, 1), (10, 1), (10, 10), (-10, 10)],
            [(-10,-10), (0, -10), (0, 10), (-10, 10)]
        ]

        border_clipping_regions_polys = [Polygon(el) for el in border_clipping_regions_polys]

        for r in vor.regions:
            poly_region = np.array(vor.vertices[r])
            if np.any((poly_region > 1.0) | (poly_region < 0.0)):
                poly_region_sh = shapely.geometry.Polygon(poly_region)
                for poly in border_clipping_regions_polys:
                    poly_region_sh = poly_region_sh.difference(poly)
                # final_regions.append(np.array(poly_region_sh.exterior.coords[:-1]))
                final_regions.append(poly_region_sh)
            else:
                # final_regions.append(np.array(vor.vertices[r]))
                final_regions.append(shapely.geometry.Polygon(poly_region))

        graph = Graph(num_nodes=len(final_regions))
        for i in range(len(final_regions)):
            graph.add_node(i)
        for (p1, p2) in all_neighbours.tolist():
            graph.add_edge(p1, p2)

        start = time.time()
        poly_count_after_merges = random.randint(self.cfg.min_num_of_polygons_after_merge, self.cfg.max_num_of_polygons_after_merge)
        poly_count_after_merges = min(poly_count_after_merges, len(final_regions))

        node_to_poly_idx = {i : [i] for i in range(len(final_regions))}
        while len(graph.get_all_nodes()) != poly_count_after_merges:
            # if len(graph.get_all_nodes()) <= self.min_perc_of_final_shapes * num_of_initial_shapes or len(
            #         graph.get_all_nodes()) == 2:
            #     break
            n1 = random.choice(graph.get_all_nodes())
            n2 = random.choice(graph.get_neighbours(n1))
            added_node = graph.merge_nodes(n1, n2)
            node_to_poly_idx[added_node] = node_to_poly_idx[n1] + node_to_poly_idx[n2]

        end = time.time()
        print("Time consumed in merging nodes: ", end - start)

        # start = time.time()
        resulting_polys = []
        for n in graph.get_all_nodes():
            node_polys = node_to_poly_idx[n]
            all_polys = [final_regions[n] for n in node_polys]
            resulting_polys.append(unary_union(all_polys))

        multi_polygons = [p for p in resulting_polys if p.type == 'MultiPolygon']
        simple_polygons = [p for p in resulting_polys if p.type != 'MultiPolygon']
        for p in multi_polygons:
            for g in p.geoms:
                simple_polygons.append(g)
        self.current_polys = simple_polygons

        # for polygon1 in resulting_polys:
        #     x, y = polygon1.exterior.xy
        #     plt.plot(x, y)
        # plt.show()

        self.current_polys = [Polygon(p) for p in self.current_polys]
        self.current_polys = [
            affinity.scale(p, xfact=self.cfg.raster_width, yfact=self.cfg.raster_height, origin=(0, 0, 0)) for p in
            self.current_polys]

        self.draw_state()

    def get_ideal_actions(self):
        initial_polys_left_bottom = [rasterio.features.bounds(poly)[:2] for poly in self.initial_polys]
        current_polys_left_bottom = [rasterio.features.bounds(poly)[:2] for poly in self.current_polys]
        resulting_actions = []
        for (ip_left, ip_bottom), (cp_left, cp_bottom) in zip(initial_polys_left_bottom, current_polys_left_bottom):
            resulting_actions.append([ip_left - cp_left, ip_bottom - cp_bottom])

        return resulting_actions

if __name__ == "__main__":

    cfg = NestEnvConfig()
    ne = NestingEnvironment(cfg)
    for _ in range(20):
        state = ne.reset()
        for _ in range(4):
            num_polys = len(ne.current_polys)
            actions = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(num_polys)]
            ne.step(actions)
        ideal_actions = ne.get_ideal_actions()
        ne.step(ideal_actions)



