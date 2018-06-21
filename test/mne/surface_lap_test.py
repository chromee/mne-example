import numpy as np
from scipy import special
import math
import mne
import pylab as plt
import SurfaceLaplacian.surface_laplacian

data = np.genfromtxt('mne/SurfaceLaplacian/example data.csv', delimiter=',') # load the data
data = data.reshape((64, 640, 99), order='F') # re-arrange data into a 3d array
data = np.rollaxis(data, 2) # swap data's shape
data = (data)*1e-6 # re-scale data
coordinates = np.genfromtxt('mne/SurfaceLaplacian/coordinates.csv', delimiter=',') # get electrode positions

ch_names = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1',
            'P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz',
            'F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2',
            'P4','P6','P8','P10','PO8','PO4','O2'] # channel names

sfreq = 256 # sampling rate

pos = np.rollaxis(coordinates, 1) # swap coordinates' shape
pos[:,[0]], pos[:,[1]] = pos[:,[1]], pos[:,[0]] # swap coordinates' positions
pos[:,[0]] = pos[:,[0]] * -1 # invert first coordinate
dig_ch_pos = dict(zip(ch_names, pos)) # assign channel names to coordinates
montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos) # make montage

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg', montage=montage) # create info

epochs = mne.EpochsArray(data=data, info=info, tmin=-1) # make Epochs onject


surf_lap, surf_orig = SurfaceLaplacian.surface_laplacian.surface_laplacian(epochs=epochs, m=4, leg_order=50, smoothing=1e-5, montage=montage)

bf_erp = surf_orig.average()
at_erp = surf_lap.average()

bf_erp.plot_topomap(np.arange(0, 0.7, 0.1), vmin=-10, vmax=10, units="$\mu V$", time_unit='ms', cmap="jet", title="Voltage", scalings=dict(eeg=1e6))
at_erp.plot_topomap(np.arange(0, 0.7, 0.1), vmin=-40, vmax=40, units="$\mu V/mm^2$", time_unit='ms', cmap="jet", title="Laplacian", scalings=dict(eeg=2e6))
plt.close()