import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import matplotlib.gridspec as grd
import numpy as np

from tools.polygon_selector import make_onselect_image, PolygonSelectionHandler
from tools.hsi_to_rgb import hsi_to_rgb

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class PolygonSelectionHandler:
    def __init__(self, img_rgb, img_spectra, wl_cal, id, target_axs):
        self.img_rgb = img_rgb
        self.mask = None
        self.indices = None
        self.img_spectra = img_spectra
        self.wl_cal = wl_cal
        self.target_axs = target_axs
        self.avg_spectrum = None
        self.id = id
        self.figure = target_axs.figure

    def onselect(self, verts):
        path = Path(verts)
        ny, nx = self.img_rgb.shape[:2]
        xg, yg = np.meshgrid(np.arange(nx), np.arange(ny))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        mask = path.contains_points(coords).reshape(ny, nx)

        self.mask = mask
        self.indices = np.argwhere(mask)

        self.avg_spectrum = np.mean(self.img_spectra[mask], 0)
        avg_spectrum_list[self.id] = self.avg_spectrum

        for art in list(self.target_axs.lines):
            art.remove()

        for i in range(2):
            self.target_axs.plot(
                self.wl_cal, avg_spectrum_list[i], color=default_colors[i]
            )
        self.figure.canvas.draw_idle()
        print(f"Image pixels selected: {np.sum(mask)}")


matfile_in = io.loadmat(r"aviva_eye_aligned_copy.mat")
wl_cal = np.squeeze(matfile_in["Wavelengths"])
lr = matfile_in["I_MS_LR"]
lr_rgb = hsi_to_rgb(lr, wl_cal)

matfile_out = io.loadmat(
    r"Outputs\R-PNN\20250612_143239\aviva_eye_aligned_copy_R-PNN.mat"
)
sr = np.squeeze(matfile_out["I_MS"])
sr = np.transpose(sr, (1, 2, 0))
sr_rgb = hsi_to_rgb(sr, wl_cal)


avg_spectrum_list = [np.zeros_like(wl_cal), np.zeros_like(wl_cal)]

fig, axs = plt.subplots(1, 3)
axs[0].imshow(lr_rgb)
axs[1].imshow(sr_rgb)
for i in range(2):
    axs[2].plot(wl_cal, avg_spectrum_list[i])
axs[2].set_title("Spectra")

handler_lr = PolygonSelectionHandler(lr_rgb, lr, wl_cal, 0, axs[2])
polysel_lr = PolygonSelector(axs[0], handler_lr.onselect)

handler_sr = PolygonSelectionHandler(sr_rgb, sr, wl_cal, 1, axs[2])
polysel_sr = PolygonSelector(axs[1], handler_sr.onselect)

plt.show()
