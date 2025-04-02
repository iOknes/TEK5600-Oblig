import numpy as np
import matplotlib.pyplot as plt
from read_data import read_h5, read_h5_yx

fpath_metsim = "data/metsim1_2d.h5"
fpath_isabel = "data/isabel_2d.h5"

u_m, v_m = read_h5(fpath_metsim)
u_i, v_i = read_h5_yx(fpath_isabel)

x_m, y_m = np.meshgrid(*(np.linspace(0,1,i) for i in u_m.shape))
x_i, y_i = np.meshgrid(*(np.linspace(0,1,i) for i in u_i.shape))

fig, ax = plt.subplots(1,2)

ax[0].set_title("Metsim")
ax[0].invert_yaxis()
nm = 4
#ax[0].quiver(x_m[::nm,::nm], y_m[::nm,::nm], u_m[::nm,::nm], v_m[::nm,::nm])
ax[0].quiver(u_m[::nm,::nm], v_m[::nm,::nm])

ax[1].set_title("Isabel")
ax[1].invert_yaxis()
ni = 20
#ax[1].quiver(x_i[::ni,::ni], y_i[::ni,::ni], u_i[::ni,::ni], v_i[::ni,::ni])
ax[1].quiver(u_i[::ni,:1:ni], v_i[::ni,:1:ni], scale=1000)

plt.show()
