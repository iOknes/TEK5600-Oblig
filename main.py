import matplotlib.pyplot as plt

from read_data import read_h5_yx
from field_line import generate_field_lines, plot_field_lines
from lic import lic, noise_texture, plot_lic


outpath = "output/isabel"
ui, vi = read_h5_yx("data/isabel_2d.h5")

# Metsim
# Field lines - seeding strategies
#fig, ax = plt.subplots(1,3)


# Fiel line lengths
# Field line - dx and integrator
# LI integrator
# LIC line/filter length

# Isabel
def field_line_seeding_strats():
    # Field lines - seeding strategies
    fig, ax = plt.subplots(1,3, figsize=(24,9))
    fli_uni = generate_field_lines(ui, vi, 0.2, 20, seeding="uniform")
    fli_ran = generate_field_lines(ui, vi, 0.2, 400, seeding="random")
    fli_vor = generate_field_lines(ui, vi, 0.2, 20, seeding="vorticity")
    plot_field_lines(ax[0], fli_uni)
    ax[0].set_title("Isabel uniform seeding")
    ax[0].invert_yaxis()
    plot_field_lines(ax[1], fli_ran)
    ax[1].set_title("Isabel random seeding")
    ax[1].invert_yaxis()
    plot_field_lines(ax[2], fli_vor)
    ax[2].set_title("Isabel vorticity seeding")
    ax[2].invert_yaxis()
    plt.savefig(f"{outpath}_seeding")


def field_line_seed_density():
    # Seed point density
    fig, ax = plt.subplots(2,3, figsize=(24,18))
    fli_uni_10 = generate_field_lines(ui, vi, 0.5, 10, seeding="uniform")
    fli_uni_20 = generate_field_lines(ui, vi, 0.5, 20, seeding="uniform")
    fli_ran_100 = generate_field_lines(ui, vi, 0.2, 100, seeding="random")
    fli_ran_400 = generate_field_lines(ui, vi, 0.2, 400, seeding="random")
    fli_vor_10 = generate_field_lines(ui, vi, 0.2, 10, seeding="vorticity")
    fli_vor_20 = generate_field_lines(ui, vi, 0.2, 20, seeding="vorticity")

    plot_field_lines(ax[0,0], fli_uni_10)
    ax[0,0].set_title("Isabel - Uniform 100 points")
    ax[0,0].invert_yaxis()

    plot_field_lines(ax[1,0], fli_uni_20)
    ax[1,0].set_title("Isabel - Uniform 400 points")
    ax[1,0].invert_yaxis()

    plot_field_lines(ax[0,1], fli_ran_100)
    ax[0,1].set_title("Isabel - Random 100 points")
    ax[0,1].invert_yaxis()

    plot_field_lines(ax[1,1], fli_ran_400)
    ax[1,1].set_title("Isabel - Random 400 points")
    ax[1,1].invert_yaxis()

    plot_field_lines(ax[0,2], fli_vor_10)
    ax[0,2].set_title("Isabel - Vorticity 100 points")
    ax[0,2].invert_yaxis()

    plot_field_lines(ax[1,2], fli_vor_20)
    ax[1,2].set_title("Isabel - Vorticity 400 points")
    ax[1,2].invert_yaxis()

    plt.savefig(f"{outpath}_seed_density")


def field_line_lengths():
    # Field line lengths
    fig, ax = plt.subplots(1,2, figsize=(16,9))
    fli_short = generate_field_lines(ui, vi, 0.1, 20, seeding="uniform")
    fli_long = generate_field_lines(ui, vi, 0.5, 20, seeding="uniform")

    plot_field_lines(ax[0], fli_short)
    ax[0].set_title("Isabel - 0.1 line length")
    ax[0].invert_yaxis()

    plot_field_lines(ax[1], fli_long)
    ax[1].set_title("Isabel - 0.5 line length")
    ax[1].invert_yaxis()

    plt.savefig(f"{outpath}_line_length")

def field_line_integrator():
    # Field line - dx and integrator
    fig, ax = plt.subplots(2,2, figsize=(16,16))
    fli_fe_01 = generate_field_lines(ui, vi, 0.5, 20, dx=0.1, integrator="fe", seeding="uniform")
    fli_fe_1 = generate_field_lines(ui, vi, 0.5, 20, dx=1, integrator="fe", seeding="uniform")
    fli_rk4_01 = generate_field_lines(ui, vi, 0.5, 20, dx=0.1, integrator="rk4", seeding="uniform")
    fli_rk4_1 = generate_field_lines(ui, vi, 0.5, 20, dx=1, integrator="rk4", seeding="uniform")

    plot_field_lines(ax[0,0], fli_fe_01)
    ax[0,0].set_title("Isabel FE dx=0.1")
    ax[0,0].invert_yaxis()

    plot_field_lines(ax[0,1], fli_fe_1)
    ax[0,1].set_title("Isabel FE dx=1")
    ax[0,1].invert_yaxis()

    plot_field_lines(ax[1,0], fli_rk4_01)
    ax[1,0].set_title("Isabel RK4 dx=0.1")
    ax[1,0].invert_yaxis()

    plot_field_lines(ax[1,1], fli_rk4_1)
    ax[1,1].set_title("Isabel RK4 dx=1")
    ax[1,1].invert_yaxis()

    plt.savefig(f"{outpath}_integrator_dx")

def lic_integrator():
    # LIC integrator
    fig, ax = plt.subplots(1,2, figsize=(16,9))
    lic_fe = lic(ui, vi, noise_texture(*ui.shape, level=1024), line_len=0.4, integrator="fe")
    lic_rk4 = lic(ui, vi, noise_texture(*ui.shape, level=1024), line_len=0.4, integrator="rk4")

    plot_lic(ax[0], lic_fe)
    ax[0].set_title("Isabel - LIC FE")

    plot_lic(ax[1], lic_rk4)
    ax[1].set_title("Isabel - LIC RK4")

    plt.savefig(f"{outpath}_lic_integrator")

def lic_line_length():
    # LIC line/filter length
    fig, ax = plt.subplots(1,3, figsize=(16,9))
    lic_005 = lic(ui, vi, noise_texture(*ui.shape, level=1024), line_len=0.05, integrator="fe")
    lic_02 = lic(ui, vi, noise_texture(*ui.shape, level=1024), line_len=0.2, integrator="fe")
    lic_05 = lic(ui, vi, noise_texture(*ui.shape, level=1024), line_len=0.5, integrator="fe")

    plot_lic(ax[0], lic_005)
    ax[0].set_title("Isabel - LIC 0.05 line length")

    plot_lic(ax[1], lic_02)
    ax[1].set_title("Isabel - LIC 0.2 line length")

    plot_lic(ax[2], lic_05)
    ax[2].set_title("Isabel - LIC 0.5 line length")

    plt.savefig(f"{outpath}_lic_line_length")

field_line_seeding_strats()
field_line_seed_density()
field_line_lengths()
field_line_integrator()
