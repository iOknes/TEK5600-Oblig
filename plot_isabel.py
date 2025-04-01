import numpy as np
import matplotlib.pyplot as plt
from read_data import read_h5

fpath_metsim = "data/metsim1_2d.h5"
fpath_isabel = "data/isabel_2d.h5"

u_m, v_m = read_h5(fpath_metsim)
u_i, v_i = read_h5(fpath_isabel)

x_m, y_m = np.meshgrid(*(np.linspace(0,1,i) for i in u_m.shape))
x_i, y_i = np.meshgrid(*(np.linspace(0,1,i) for i in u_i.shape))

fig, ax = plt.subplots(2,2)

#ax[0,0].contourf(x_m, y_m, np.sqrt(u_m**2, v_m**2), levels=256)
#ax[0,1].contourf(x_i, y_i, np.sqrt(u_i**2, v_i**2), levels=256)
ax[0,0].set_title("Metsim X-comp")
ax[0,0].set_xlabel("x")
ax[0,0].set_ylabel("y")
c00 = ax[0,0].contourf(x_m, y_m, u_m, levels=256)
ax[1,0].set_title("Metsim Y-comp")
ax[1,0].set_xlabel("x")
ax[1,0].set_ylabel("y")
c10 = ax[1,0].contourf(x_m, y_m, v_m, levels=256)
ax[0,1].set_title("Isabel X-comp")
ax[0,1].set_xlabel("x")
ax[0,1].set_ylabel("y")
c01 = ax[0,1].contourf(x_i, y_i, u_i, levels=256)
ax[1,1].set_title("Isabel Y-comp")
ax[1,1].set_xlabel("x")
ax[1,1].set_ylabel("y")
c11 = ax[1,1].contourf(x_i, y_i, v_i, levels=256)
plt.colorbar(c00, ax=ax[0,0])
plt.colorbar(c10, ax=ax[1,0])
plt.colorbar(c01, ax=ax[0,1])
plt.colorbar(c11, ax=ax[1,1])
#ax[0].invert_xaxis()
ax[0,0].invert_yaxis()
ax[1,0].invert_yaxis()
#ax[1].invert_xaxis()
ax[0,1].invert_yaxis()
ax[1,1].invert_yaxis()
plt.show()
