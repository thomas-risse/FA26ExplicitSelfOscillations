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
t = np.arange(int(duration * sr)) / sr
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
fig = plt.figure(figsize=set_size(width='FA', height_ratio=1))
tmin = 0
tmax = 0.05
idxmin = int(tmin * sr)
idxmax = int(tmax * sr)


plt.plot(time[idxmin:idxmax], Pstored[idxmin: idxmax], label="Stored")

plt.plot(time[idxmin:idxmax], Pext[idxmin: idxmax], label="External")
# plt.plot(time[idxmin:idxmax], PextSub[idxmin: idxmax], label = r"$P_{sub}$")
# plt.plot(time[idxmin:idxmax], PextSup[idxmin: idxmax], label = r"$P_{sup}$")

plt.plot(time[idxmin:idxmax], Pdiss[idxmin: idxmax], label="Dissipated")
# plt.plot(time[idxmin:idxmax], PdissFlow[idxmin: idxmax], label = "Flow")
# plt.plot(time[idxmin:idxmax], PdissFolds[idxmin: idxmax], label = "Folds")


# plt.plot( -(qfolds[:10000, :] - h0s) * 100)

plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.legend()
plt.grid()


plt.tight_layout()
fig.savefig("/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_powers.pdf",
            bbox_inches="tight")
