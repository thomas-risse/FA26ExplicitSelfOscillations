import json
import numpy as np
import matplotlib.pyplot as plt

import h5py
import subprocess
from os import path, makedirs
from helper_plots import set_size


# Solver parameters
if not path.exists("results"):
    makedirs("results")
fname = f"results/testVoice.hdf5"

sr = 44100
duration = 1

t = np.linspace(0, duration, int(sr * duration))

# Excitation
Pin0 = 800
trise = 0.001
tdecay = 0
tend = 0
Pin = np.ones_like(t) * Pin0
Pin[:int(trise * sr)] = np.linspace(0, Pin0, int(trise * sr))

# Write file to pass to C++
with h5py.File(fname, "w") as f:
    f.attrs["sr"] = sr
    f.attrs["duration"] = duration * 1
    f["Pmouth"] = Pin

# Run simulation
subprocess.run(
    ["../build/RunLarynx", f"{path.realpath(fname)}"])

# Get Results
with h5py.File(fname, "r") as f:
    q = f["foldDisplacement"][:]
    restPositions = f["restPositions"][:]
    supGlottalFlow = f["supGlottalFlow"][:]
    meanGlottalFlow = f["meanGlottalFlow"][:]
    pressureDrop = f["pressureDrop"][:]

    time = f["time"][:]

    Pext = f["Pext"][:]
    PextSub = f["PextSub"][:]
    PextSup = f["PextSup"][:]

    Pdiss = f["Pdiss"][:]
    PdissFlow = f["PdissFlow"][:]
    PdissFolds = f["PdissFolds"][:]

    Pstored = f["Pstored"][:]

# Plots
tmin = 0.03
tmax = 0.05
idxmin = int(tmin * sr)
idxmax = int(tmax * sr)

# Powers$

fig = plt.figure(figsize=set_size(width='FA', height_ratio=0.7))

plt.plot(time[idxmin:idxmax], Pstored[idxmin: idxmax], label="Stored")

plt.plot(time[idxmin:idxmax], Pext[idxmin: idxmax], label="External")
# plt.plot(time[idxmin:idxmax], PextSub[idxmin: idxmax], label = r"$P_{sub}$")
# plt.plot(time[idxmin:idxmax], PextSup[idxmin: idxmax], label = r"$P_{sup}$")

plt.plot(time[idxmin:idxmax], Pdiss[idxmin: idxmax], label="Dissipated")
# plt.plot(time[idxmin:idxmax], PdissFlow[idxmin: idxmax], label = "Flow")
# plt.plot(time[idxmin:idxmax], PdissFolds[idxmin: idxmax], label = "Folds")

plt.xlim([tmin, tmax])

plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.legend(frameon=True, loc=1)
plt.grid()


plt.tight_layout()
fig.savefig("/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_powers.pdf",
            bbox_inches="tight")

# Masses displacements and glottal flow
fig, axs = plt.subplots(figsize=set_size(
    width='FA', height_ratio=0.5, subplots=(2, 1)), nrows=2, ncols=1, sharex=True, height_ratios=[1, 0.5])

plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axs[0].plot(time[idxmin:idxmax], (restPositions[0] -
                                  q[idxmin: idxmax, 0]) * 1000, label="Lower", ls="--")
axs[0].plot(time[idxmin:idxmax], (restPositions[1] -
                                  q[idxmin: idxmax, 1]) * 1000, label="Upper", ls="-")
axs[0].plot(time[idxmin:idxmax], (restPositions[2] -
                                  q[idxmin: idxmax, 2]) * 1000, label="Body", ls=":")
axlims = axs[0].get_ylim()
axs[0].axhspan(-1, 0, alpha=0.3, color="grey")
axs[0].set_ylim(axlims)


axs[1].plot(time, supGlottalFlow, color="red")
# axs[1].plot(time, meanGlottalFlow)


plt.xlim([tmin, tmax])
plt.xlabel("Time (s)")
axs[0].set_ylabel("Opening\n(mm)",
                  multialignment="center")
axs[1].set_ylabel("Flow\n(m$^3.$s$^{-1}$)", multialignment="center")
axs[0].legend(frameon=True, loc=1)
fig.align_ylabels()

for ax in axs:
    ax.grid()

plt.tight_layout()
fig.subplots_adjust(hspace=0.25)

fig.savefig("/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_displacements.pdf",
            bbox_inches="tight")

# If export to json is wanted for some reason
# f = h5py.File('results/testVoice.hdf5', 'r')
# out = {
#     'time': f['time'][:].tolist(),
#     'foldDisplacement': f['foldDisplacement'][:].tolist(),
#     'meanGlottalFlow': f['meanGlottalFlow'][:].tolist(),
# }
# with open('voiceData.json', 'w') as fp:
#     json.dump(out, fp)
