import matplotlib.pyplot as plt

from read_data import read_h5
from field_line import generate_field_lines, plot_field_lines
from lic import lic, noise_texture, plot_lic


outpath = "output/metsim"
ui, vi = read_h5("data/metsim1_2d.h5")


# Metsim
# Field lines - seeding strategies
fig, ax = plt.subplots(1,3, figsize=(24,9))
fli_uni = generate_field_lines(ui, vi, 0.2, 20, seeding="uniform")
fli_ran = generate_field_lines(ui, vi, 0.2, 400, seeding="random")
fli_vor = generate_field_lines(ui, vi, 0.2, 20, seeding="vorticity")
plot_field_lines(ax[0], fli_uni, metsim=True)
ax[0].set_title("Metsim uniform seeding")
ax[0].invert_yaxis()
plot_field_lines(ax[1], fli_ran, metsim=True)
ax[1].set_title("Metsim random seeding")
ax[1].invert_yaxis()
plot_field_lines(ax[2], fli_vor, metsim=True)
ax[2].set_title("Metsim vorticity seeding")
ax[2].invert_yaxis()
plt.savefig(f"{outpath}_seeding")


# Seed point density
fig, ax = plt.subplots(2,3, figsize=(24,18))
fli_uni_10 = generate_field_lines(ui, vi, 0.5, 10, seeding="uniform")
fli_uni_20 = generate_field_lines(ui, vi, 0.5, 20, seeding="uniform")
fli_ran_100 = generate_field_lines(ui, vi, 0.2, 100, seeding="random")
fli_ran_400 = generate_field_lines(ui, vi, 0.2, 400, seeding="random")
fli_vor_10 = generate_field_lines(ui, vi, 0.2, 10, seeding="vorticity")
fli_vor_20 = generate_field_lines(ui, vi, 0.2, 20, seeding="vorticity")

plot_field_lines(ax[0,0], fli_uni_10, metsim=True)
ax[0,0].set_title("Metsim - Uniform 100 points")
ax[0,0].invert_yaxis()

plot_field_lines(ax[1,0], fli_uni_20, metsim=True)
ax[1,0].set_title("Metsim - Uniform 400 points")
ax[1,0].invert_yaxis()

plot_field_lines(ax[0,1], fli_ran_100, metsim=True)
ax[0,1].set_title("Metsim - Random 100 points")
ax[0,1].invert_yaxis()

plot_field_lines(ax[1,1], fli_ran_400, metsim=True)
ax[1,1].set_title("Isabel - Random 400 points")
ax[1,1].invert_yaxis()

plot_field_lines(ax[0,2], fli_vor_10, metsim=True)
ax[0,2].set_title("Isabel - Vorticity 100 points")
ax[0,2].invert_yaxis()

plot_field_lines(ax[1,2], fli_vor_20, metsim=True)
ax[1,2].set_title("Isabel - Vorticity 400 points")
ax[1,2].invert_yaxis()

plt.savefig(f"{outpath}_seed_density")


# Field line lengths
fig, ax = plt.subplots(1,2, figsize=(16,9))
fli_short = generate_field_lines(ui, vi, 0.1, 20, seeding="uniform")
fli_long = generate_field_lines(ui, vi, 0.5, 20, seeding="uniform")
plot_field_lines(ax[0], fli_short, metsim=True)
ax[0].set_title("Metsim - 0.1 line length")
ax[0].invert_yaxis()
plot_field_lines(ax[1], fli_long, metsim=True)
ax[1].set_title("Metsim - 0.5 line length")
ax[1].invert_yaxis()
plt.savefig(f"{outpath}_line_length")

# Field line - dx and integrator
fig, ax = plt.subplots(2,2, figsize=(16,16))
fli_fe_01 = generate_field_lines(ui, vi, 0.2, 20, dx=0.1, integrator="fe", seeding="uniform")
fli_fe_1 = generate_field_lines(ui, vi, 0.5, 20, dx=1, integrator="fe", seeding="uniform")
fli_rk4_01 = generate_field_lines(ui, vi, 0.5, 20, dx=0.1, integrator="rk4", seeding="uniform")
fli_rk4_1 = generate_field_lines(ui, vi, 0.5, 20, dx=1, integrator="rk4", seeding="uniform")
plot_field_lines(ax[0,0], fli_fe_01, metsim=True)
ax[0,0].set_title("Metsim FE dx=0.1")
ax[0,0].invert_yaxis()
plot_field_lines(ax[0,1], fli_fe_1, metsim=True)
ax[0,1].set_title("Metsim FE dx=1")
ax[0,1].invert_yaxis()
plot_field_lines(ax[1,0], fli_rk4_01, metsim=True)
ax[1,0].set_title("Metsim RK4 dx=0.1")
ax[1,0].invert_yaxis()
plot_field_lines(ax[1,1], fli_rk4_1, metsim=True)
ax[1,1].set_title("Metsim RK4 dx=1")
ax[1,1].invert_yaxis()

plt.savefig(f"{outpath}_integrator_dx")

# LIC integrator
fig, ax = plt.subplots(1,2, figsize=(16,9))
lic_fe = lic(ui, vi, noise_texture(*ui.shape), integrator="fe")
lic_rk4 = lic(ui, vi, noise_texture(*ui.shape), integrator="rk4")
plot_lic(ax[0], lic_fe, metsim=True)
ax[0].set_title("Metsim - LIC FE")
ax[0].invert_yaxis()
plot_lic(ax[1], lic_rk4, metsim=True)
ax[1].set_title("Metsim - LIC RK4")
ax[1].invert_yaxis()
plt.savefig(f"{outpath}_lic_integrator")

# LIC line/filter length
fig, ax = plt.subplots(2,2, figsize=(16,16))
lic_02 = lic(ui, vi, noise_texture(*ui.shape), line_len=0.2, integrator="fe")
lic_03 = lic(ui, vi, noise_texture(*ui.shape), line_len=0.3, integrator="fe")
lic_04 = lic(ui, vi, noise_texture(*ui.shape), line_len=0.4, integrator="fe")
lic_05 = lic(ui, vi, noise_texture(*ui.shape), line_len=0.5, integrator="fe")

plot_lic(ax[0,0], lic_02, metsim=True)
ax[0,0].set_title("Metsim - LIC 0.2 line length")
ax[0,0].invert_yaxis()

plot_lic(ax[0,1], lic_03, metsim=True)
ax[0,1].set_title("Metsim - LIC 0.3 line length")
ax[0,1].invert_yaxis()

plot_lic(ax[1,0], lic_04, metsim=True)
ax[1,0].set_title("Metsim - LIC 0.4 line length")
ax[1,0].invert_yaxis()

plot_lic(ax[1,1], lic_05, metsim=True)
ax[1,1].set_title("Metsim - LIC 0.5 line length")
ax[1,1].invert_yaxis()

plt.savefig(f"{outpath}_lic_line_length")

#output = lic(u, v, noise_texture(*(u.shape)), integrator=forward_euler)
#plt.imshow(output.T, cmap="grey")
#plt.show()
