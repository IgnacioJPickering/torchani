import torchani
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def bare_angular_cosine(theta, zeta, shift_z, norm=True):
    theta = np.where(theta < np.pi, theta, 2 * np.pi - theta )
    if norm:
        norm = 1 / (1.5** zeta)
    else:
        norm = 1
    return norm * (1 + 0.5 * np.cos(theta - shift_z)) ** zeta

def bare_angular_terms(x, theta, eta_a, shift_a, zeta, shift_z, norm=True, double=False, scale=1):
    if double:
        factor = 2
    else:
        factor = 1
    return factor * bare_gaus(x, eta_a, shift_a, scale=scale)* bare_angular_cosine(theta, zeta, shift_z, norm=norm)



r_radial_dummy = np.linspace(0.0, radial_cutoff + 0.5, 1000)
r_angular_dummy = np.linspace(0.0, angular_cutoff + 0.5, 1000)
angles = np.linspace(0.0, 2 * np.pi, 1000)

r_angular_mesh, angles_mesh = np.meshgrid(r_angular_dummy, angles)

fig, ax = plt.subplots()
for eta, shift, s in zip(radial_etas, radial_shifts, scales):
    ax.plot(r_radial_dummy, bare_gaus(r_radial_dummy, eta, shift, scale=s))
ax.vlines(x=[radial_shifts[0], radial_cutoff], ymin=0, ymax=1.3)
ax.vlines(x=radial_shifts, ymin=0, ymax=1.0, colors='k', linestyles='dashed')
ax.set_xlim(0.0, radial_cutoff + 0.5)
ax.set_title('Radial terms')
plt.show()

fig, ax = plt.subplots()
for eta, shift, s in zip(radial_etas, radial_shifts, scales):
    ax.plot(r_radial_dummy, bare_gaus(r_radial_dummy, eta, shift, scale=s) *
            cosine_cutoff(r_radial_dummy, radial_cutoff))
ax.vlines(x=[radial_shifts[0], radial_cutoff], ymin=0, ymax=1.3)
ax.vlines(x=radial_shifts, ymin=0, ymax=1.0, colors='k', linestyles='dashed')
ax.set_xlim(0.0, radial_cutoff + 0.5)
ax.set_title('Radial terms + cutoff')
plt.show()

fig, ax = plt.subplots()
for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
    ax.plot(r_angular_dummy, bare_gaus(r_angular_dummy, eta, shift, scale=s))
ax.vlines(x=[angular_shifts[0], angular_cutoff], ymin=0, ymax=1.3)
ax.set_xlim(0.0, angular_cutoff + 0.5)
ax.set_title('Angular terms')
plt.show()

fig, ax = plt.subplots()
for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
    A = bare_gaus(r_angular_dummy, eta, shift, scale=s) * cosine_cutoff(r_angular_dummy, angular_cutoff) ** 2
    ax.plot(r_angular_dummy, A)
ax.vlines(x=[angular_shifts[0], angular_cutoff], ymin=0, ymax=1.3)
ax.set_xlim(0.0, angular_cutoff + 0.5)
ax.set_title('Angular terms + cutoff')
plt.show()

fig, ax = plt.subplots()
for zeta, shift_z in zip(zetas, angle_sections):
    ax.plot(angles, bare_angular_cosine(angles, zeta, shift_z) )
#ax.vlines([0.0, np.pi, 2 * np.pi], ymin=0, ymax=1.3)
d = 8
assert d % 2 == 0
ax.set_xticks(np.linspace(0, 2* np.pi, d + 1))
#ax.set_xticklabels(['0'] +  [r'$\frac{'  f'{j}'  r'}{' f'{d//2}' r'}\pi$' for j in range(1, d + 1)])
ax.set_xlim(-0.01, 2 * np.pi + 0.01)
ax.set_title('Angle sections')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(projection='polar')
build = True
colormap_plot = []
for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
    for zeta, shift_z in zip(zetas, angle_sections):
        z = bare_angular_terms(r_angular_mesh, angles_mesh, eta, shift, zeta, shift_z, norm=True, scale=s)
        colormap_plot.append(z)
        ax.contour(angles_mesh, r_angular_mesh, z, levels=3)
colormap_plot = np.asarray(colormap_plot).max(axis=0)
mappable = ax.pcolormesh(angles_mesh, r_angular_mesh, colormap_plot, cmap = 'jet')
fig.colorbar(mappable)
ax.set_title('Polar angular')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
build = True
colormap_plot = []
for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
    for zeta, shift_z in zip(zetas, angle_sections):
        z = bare_angular_terms(r_angular_mesh, angles_mesh, eta, shift, zeta, shift_z, norm=True, scale=s) * cosine_cutoff(r_angular_dummy, angular_cutoff) ** 2
        colormap_plot.append(z)
        ax.contour(angles_mesh, r_angular_mesh, z, levels=3)
colormap_plot = np.asarray(colormap_plot).max(axis=0)
mappable = ax.pcolormesh(angles_mesh, r_angular_mesh, colormap_plot, cmap = 'jet')
fig.colorbar(mappable)
ax.set_title('Polar angular + cutoff')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
build = True
colormap_plot = []
for eta, shift, s in zip(angular_etas, angular_shifts, scales_angular):
    for zeta, shift_z in zip(zetas, angle_sections):
        z = bare_angular_terms(r_angular_mesh, angles_mesh, eta, shift, zeta, shift_z, norm=True, scale=s) * cosine_cutoff(r_angular_dummy, angular_cutoff) ** 2
        colormap_plot.append(z)
        #ax.contour(angles_mesh, r_angular_mesh, z, levels=3)
X, Y = r_angular_mesh * np.cos(angles_mesh), r_angular_mesh * np.sin(angles_mesh)
colormap_plot = np.asarray(colormap_plot).max(axis=0)
mappable = ax.plot_surface(X, Y, colormap_plot, cmap = 'jet')
fig.colorbar(mappable)
ax.set_title('Polar angular + cutoff')
plt.show()
exit()


