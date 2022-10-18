import random

import shapely.geometry
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

def cut_polygon_by_lines(polygon, line):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)

def simple_voronoi(vor, saveas=None, lim=None):
    # Make Voronoi Diagram
    fig = voronoi_plot_2d(vor, show_points=True, show_vertices=True, s=4)

    # Configure figure
    fig.set_size_inches(5,5)
    plt.axis("equal")

    if lim:
        plt.xlim(*lim)
        plt.ylim(*lim)

    if not saveas is None:
        plt.savefig("%s.png"%saveas)

    plt.show()

class NestEnvConfig:
    def __init__(self):
        self.min_pts = 200
        self.max_pts = 2000

        self.border_min_pts = 1
        self.border_max_pts = 1


class NestingEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        self.build_state()

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

    def build_state(self):
        num_pts = random.randint(self.cfg.min_pts, self.cfg.max_pts)
        pts = np.random.random((num_pts, 2))
        border_pts = self.get_bordering_pts()
        border_pad_pts = self.get_border_padding_pts()
        all_pts = np.concatenate([pts, border_pts, border_pad_pts], axis=0)

        vor = Voronoi(all_pts, incremental=True)
        # invalid_pts = np.any((vor.vertices > 1.0) | (vor.vertices < 0.0), axis=-1)
        vor.regions = list(filter(lambda el: all(list(map(lambda v: v != -1, el))), vor.regions))
        all_polys = []

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
                final_regions.append(np.array(poly_region_sh.exterior.coords[:-1]))
            else:
                final_regions.append(np.array(vor.vertices[r]))

        image = Image.new("RGB", (1200, 1200))

        draw = ImageDraw.Draw(image)
        for r in final_regions:
            r *= 1000.0
            r += 100
            draw.polygon((r.flatten().tolist()), outline=200)

        image.show()

if __name__ == "__main__":

    cfg = NestEnvConfig()
    ne = NestingEnvironment(cfg)
    for _ in range(20):
        ne.reset()

