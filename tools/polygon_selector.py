import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


# TODO delete comments and only keep the class

# fig, (ax_scatter, ax_img) = plt.subplots(1, 2, figsize=(10, 5))

# x, y = np.random.rand(100), np.random.rand(100)
# img = np.random.rand(50, 50, 3)

# pts = ax_scatter.scatter(x=x, y=y)

# ax_img.imshow(img)


# def make_onselect_scatter(collection):  # pass collection to build the onselect scatter
#     # verts is list of tuples with coordinates of selected points
#     def onselect_scatter(verts):
#         path = Path(verts)
#         points = pts.get_offsets()
#         ind = np.nonzero(path.contains_points(points))[0]
#         print("Scatter selected indices:", ind)
#         # Optional: highlight selected points
#         colors = np.array(["blue"] * len(points))
#         colors[ind] = "red"
#         pts.set_color(colors)
#         fig.canvas.draw_idle()

#     return onselect_scatter


# alternatively
# class SelectFromScatter:
#     def __init__(self, ax, pts):
#         self.pts = pts
#         self.selector = PolygonSelector(ax, self.onselect)

#     def onselect(self, verts):
#         points = self.pts.get_offsets()
#         ...


def make_onselect_image(img):
    def onselect_image(verts):
        path = Path(verts)
        ny, nx = img.shape[:2]
        xg, yg = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        mask = path.contains_points(coords).reshape(ny, nx)
        print(f"Image pixels selected: {np.sum(mask)}")
        # Optional: mask image outside polygon
        # img_masked = img.copy()
        # img_masked[~mask] = 0
        # ax_img.imshow(img_masked)
        # fig.canvas.draw_idle()

    return onselect_image


# instantiation of the PolygonSelector class
# onselect_scatter = make_onselect_scatter(pts)
# selector_scatter = PolygonSelector(ax_scatter, onselect_scatter)

# onselect_image = make_onselect_image(img)
# selector_img = PolygonSelector(ax_img, onselect_image)


# plt.show()


class PolygonSelectionHandler:
    def __init__(self, img):
        self.img = img
        self.mask = None
        self.indices = None

    def onselect(self, verts):
        path = Path(verts)
        ny, nx = self.img.shape[:2]
        xg, yg = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        mask = path.contains_points(coords).reshape(ny, nx)

        self.mask = mask
        self.indices = np.argwhere(mask)

        print(f"Image pixels selected: {np.sum(mask)}")


# Later:
#   handler.mask
#   handler.indices
