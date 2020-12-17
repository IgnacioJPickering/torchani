import torchani
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa
import numpy as np
from numpy import random 

aev_computer = torchani.TemplateModel.like_ani1x().aev_computer

# these scales work for radial terms, but not for
# angular terms necessarily

# these scales were obtained by rescaling the radial terms so that they
# go between 0 and 1
scales = np.asarray([
0.9282837113533762, 
0.8810616615689186, 
0.823926001674413, 
0.7584146320997143, 
0.68612829719817, 
0.6089622857973057, 
0.5290817141020113, 
0.4484833462374985, 
0.36930025215334017, 
0.29361443294889067, 
0.22345529875946749, 
0.1606157294729754, 
0.10676448215557516, 
0.06328044533463194, 
0.03121678259062526, 
0.011104079116615408])

scales_angular = np.asarray([
0.7274285152554174, 
0.3704739662625683, 
0.11249284672261844, 
0.0144858545390771])
scales_angular = np.ones_like(scales_angular)
scales = np.ones_like(scales)

radial_cutoff = aev_computer.Rcr.item()
angular_cutoff = aev_computer.Rca.item()
radial_etas = aev_computer.EtaR.numpy().reshape(-1)
angular_etas = aev_computer.EtaA.numpy().reshape(-1)
radial_shifts = aev_computer.ShfR.numpy().reshape(-1)
angular_shifts = aev_computer.ShfA.numpy().reshape(-1)
#angle_sections = aev_computer.ShfZ.numpy().reshape(-1)
angle_sections = np.linspace(0, np.pi, 8)
zetas = aev_computer.Zeta.numpy().reshape(-1)

if len(radial_etas) != len(radial_shifts):
    radial_etas = radial_etas.repeat(len(radial_shifts))

assert len(scales) == len(radial_shifts)

if len(angular_etas) != len(angular_shifts):
    angular_etas = angular_etas.repeat(len(angular_shifts))

if len(zetas) != len(angle_sections):
    zetas = zetas.repeat(len(angle_sections))

def cosine_cutoff(x, cut):
    return 0.5 * (1 + np.cos(x * (np.pi/cut)))

def bare_gaus(x, eta, shift, scale=1):
    
    return (1/scale) * np.exp(-eta * (x - shift) ** 2)

def bare_angular_cosine(theta, zeta, shift_z, norm=False):
    theta = np.where(theta < np.pi, theta, 2 * np.pi - theta )
    if norm:
        norm = 1 / (1.5** zeta)
    else:
        norm = 1
    return norm * (1 + 0.5 * np.cos(theta - shift_z)) ** zeta

def bare_angular_terms(x, theta, eta_a, shift_a, zeta, shift_z, norm=False, double=True, scale=1):
    if double:
        factor = 2
    else:
        factor = 1
    return factor * bare_gaus(x, eta_a, shift_a, scale=scale)* bare_angular_cosine(theta, zeta, shift_z, norm=norm)

r_radial_dummy = np.linspace(0.0, radial_cutoff + 0.5, 1000)
r_angular_dummy = np.linspace(0.0, angular_cutoff + 0.5, 1000)
angles = np.linspace(0.0, 2 * np.pi, 1000)

r_angular_mesh, angles_mesh = np.meshgrid(r_angular_dummy, angles)



def plot_gausians(dummy, etas, shifts, scales, cutoff, comment, with_cut=False, with_cut_sq=False):
    fig, ax = plt.subplots()
    for eta, shift, s in zip(etas, shifts, scales):
        if with_cut:
            z = bare_gaus(dummy, eta, shift, scale=s) * cosine_cutoff(dummy, cutoff)
        elif with_cut_sq:
            z = bare_gaus(dummy, eta, shift, scale=s) * (cosine_cutoff(dummy, cutoff) ** 2)
        else:
            z = bare_gaus(dummy, eta, shift, scale=s)
        ax.plot(dummy, z)
    ax.vlines(x=[shifts[0], cutoff], ymin=0, ymax=1.3)
    ax.vlines(x=shifts, ymin=0, ymax=1.0, colors='k', linestyles='dashed')
    ax.set_xlim(0.0, cutoff + 0.5)
    plt.show()

def plot_angular_cosine(dummy, zetas, shifts, comment):
    fig, ax = plt.subplots()
    for zeta, shift_z in zip(zetas, angle_sections):
        ax.plot(angles, bare_angular_cosine(angles, zeta, shift_z) )
    d = 8
    assert d % 2 == 0
    ax.set_xticks(np.linspace(0, 2* np.pi, d + 1))
    ax.set_xlim(-0.01, 2 * np.pi + 0.01)
    ax.set_title(comment)
    plt.show()

def plot_polar(r_angular_mesh, angles_mesh, etas, shifts, zetas, angle_sections, scales, comment, cutoff, with_cut_sq=False, contours=False, surface=False):
    if surface:
        assert not contours
    fig = plt.figure()
    if surface:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot(projection='polar')
    colormap_plot = []
    for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
        for zeta, shift_z in zip(zetas, angle_sections):
            if with_cut_sq:
                z = bare_angular_terms(r_angular_mesh, angles_mesh, eta, shift, zeta, shift_z, scale=s) * cosine_cutoff(r_angular_mesh, cutoff) ** 2
            else:
                z = bare_angular_terms(r_angular_mesh, angles_mesh, eta, shift, zeta, shift_z, scale=s)
            colormap_plot.append(z)
            if contours:
                ax.contour(angles_mesh, r_angular_mesh, z, levels=3)
    colormap_plot = np.asarray(colormap_plot).max(axis=0)
    if surface:
        X, Y = r_angular_mesh * np.cos(angles_mesh), r_angular_mesh * np.sin(angles_mesh)
        mappable = ax.plot_surface(X, Y, colormap_plot, cmap = 'jet')
    else:
        mappable = ax.pcolormesh(angles_mesh, r_angular_mesh, colormap_plot, cmap = 'jet')
    fig.colorbar(mappable)
    ax.set_title('Polar angular')
    plt.show()

def simulate_histogram_gaussian(trials, cutoff, eta, shift, with_cut=False, with_cut_sq=False):
    trials = random.uniform(0.0, cutoff, trials)
    if with_cut:
        values = bare_gaus(trials, eta, shift) * cosine_cutoff(trials, cutoff)
    elif with_cut_sq:
        values = bare_gaus(trials, eta, shift) * cosine_cutoff(trials, cutoff) ** 2
    else:
        values = bare_gaus(trials, eta, shift)
    values = values[values.nonzero()]
    fig, ax = plt.subplots()
    ax.hist(values, bins=100)
    plt.show()

def simulate_histogram_cosines(trials, cutoff, zeta, shift_z):
    trials = random.uniform(0.0, cutoff, trials)
    values = bare_angular_cosine(trials, zeta, shift_z)
    values = values[values.nonzero()]
    fig, ax = plt.subplots()
    ax.hist(values, bins=100)
    plt.show()

def simulate_histogram_angles(trials, cutoff, zeta, shift_z, eta, shift):
    trials = random.uniform(0.0, cutoff, trials)
    values = bare_angular_cosine(trials, zeta, shift_z) * bare_gaus(trials, eta, shift) * cosine_cutoff(trials, cutoff) ** 2
    values = values[values.nonzero()]
    fig, ax = plt.subplots()
    ax.hist(values, bins=100)
    plt.show()

for zeta, shift_z in zip(zetas, angle_sections):
    for eta, shift in zip(angular_etas, angular_shifts):
        simulate_histogram_angles(10000, angular_cutoff, zeta, shift_z, eta, shift)

for zeta, shift in zip(zetas, angle_sections):
    simulate_histogram_cosines(100000, angular_cutoff, zeta, shift)

for eta, shift in zip(radial_etas, radial_shifts):
    simulate_histogram_gaussian(100000, radial_cutoff, eta, shift)


exit()
# radial
plot_gausians(r_radial_dummy, radial_etas, radial_shifts, scales, radial_cutoff, 'Radial terms')
# radial with cut
plot_gausians(r_radial_dummy, radial_etas, radial_shifts, scales, radial_cutoff, 'Radial terms + cutoff', with_cut=True)
# angular
plot_gausians(r_angular_dummy, angular_etas, angular_shifts, scales_angular, angular_cutoff, 'Angular terms')
# angular with cut
plot_gausians(r_angular_dummy, angular_etas, angular_shifts, scales_angular, angular_cutoff, 'Angular terms + cutoff', with_cut_sq=True)
# angular cosines
plot_angular_cosine(angles, zetas, angle_sections, 'Angular cosines')
# polar angular
plot_polar(r_angular_mesh, angles_mesh, angular_etas, angular_shifts, zetas, angle_sections, scales_angular, 'Angular polar', angular_cutoff)
# polar angular with cut
plot_polar(r_angular_mesh, angles_mesh, angular_etas, angular_shifts, zetas, angle_sections, scales_angular, 'Angular polar + cutoff', angular_cutoff, with_cut_sq=True)
# plot angular in surface form
plot_polar(r_angular_mesh, angles_mesh, angular_etas, angular_shifts, zetas, angle_sections, scales_angular, 'Angular polar + cutoff', angular_cutoff, surface=True)
# plot angular with cut in surface form
plot_polar(r_angular_mesh, angles_mesh, angular_etas, angular_shifts, zetas, angle_sections, scales_angular, 'Angular polar + cutoff', angular_cutoff, with_cut_sq=True, surface=True)




