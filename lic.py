import numpy as np

from field_line import field_line
from integration import forward_euler

from tqdm import tqdm

def noise_texture(nx, ny, level=255):
    return (level * np.random.rand(nx, ny)).astype(int)

def convolute():
    pass

def lic(ux, uy, image: np.ndarray, line_len=0.5, integrator="fe"):
    output = np.zeros(image.shape)
    fll = line_len * np.average(ux.shape)
    lx, ly = ux.shape
    tqdm_prog = tqdm(total=len(image))
    for i, row in enumerate(image):
        for j, col in enumerate(row):
            p = [round(i * image.shape[0] / ux.shape[0]), round(j * image.shape[1] / ux.shape[1])]
            fl = field_line(ux, uy, p, fll, dx=0.5, integrator=integrator, backwards=True, normalise=True)
            fl_coords = np.clip(np.round(fl).astype(int), 0, image.shape[0]-1)
            output[i,j] = np.mean(image[fl_coords])
        tqdm_prog.update()
    output = (output - output.min())
    return output / output.max()

def plot_lic(ax, lic_image, metsim=False):
    if metsim:
        ax.imshow(lic_image.T, cmap="grey")
    else:
        ax.imshow(lic_image, cmap="grey")
