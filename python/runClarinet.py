
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import h5py
import subprocess
from os import path, makedirs
from helper_plots import set_size


# ──────────────────────────────────────────────────────────────────────────────
# General setting
# ──────────────────────────────────────────────────────────────────────────────

# Input mouth pressure ramp
PsubMax = 2000  # Pa
trise = 0.01  # s

# Numerical setting
duration = 0.1  # s
sr0 = 44100  # Hz, base samplerate
# Number of higher samplerates for convergence. The max samplerate is sr0 * 2^{Nsrs}
Nsrs = 6
idx_plot = 0  # Index of the samplerate for wich the displacement and flow figures are exported. 0 for sr0, Nsrs-1 for reference

# Time interval for plotting and error computation
tmin, tmax = 0., 0.1

# Directories
result_folder = "results"  # Simulation results
figure_folder = "figure"  # Figure export
fig_width = 'FA'  # In inches, or "FA" for linewidth of Forum Acusticum template

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────


def inputPressure(t):
    return PsubMax * (t >= trise) + PsubMax * (t < trise) * t / trise


def run_simulation(sr: int, duration: float, fname: str) -> dict:
    """Write inputs, run the C++ solver, and return the relevant arrays."""
    t = np.linspace(0, duration, int(sr * duration))

    # Excitation
    Pin = inputPressure(t)

    with h5py.File(fname, "w") as f:
        f.attrs["sr"] = sr
        f.attrs["duration"] = duration
        f["Pmouth"] = Pin

    subprocess.run(
        ["../build/RunSingleReed", f"{path.realpath(fname)}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with h5py.File(fname, "r") as f:
        return {
            "q":               f["reedDisplacement"][:],
            "layPosition":     f.attrs["layPosition"],
            "resonatorFlow":   f["resonatorFlow"][:],
            "meanFlow":        f["meanFlow"][:],
            "time":            f["time"][:],
            "Pext":            f["Pext"][:],
            "PextSub":         f["PextSub"][:],
            "PextSup":         f["PextSup"][:],
            "Pdiss":           f["Pdiss"][:],
            "PdissFlow":       f["PdissFlow"][:],
            "PdissReed":       f["PdissReed"][:],
            "Pstored":         f["Pstored"][:],
            "PstoredKinetic":         f["PstoredKinetic"][:],
            "PstoredPotential":         f["PstoredPotential"][:],
            "pressureDrop":         f["pressureDrop"][:],
            "radiatedPressure":         f["radiatedPressure"][:],
        }


def l2_norm(a: np.ndarray) -> float:
    """Discrete L2 norm (RMS-like, normalised by length)."""
    return np.sqrt(np.mean(a ** 2))


if not path.exists(result_folder):
    makedirs(result_folder)


# ──────────────────────────────────────────────────────────────────────────────
# Reference simulation (highest sample rate)
# ──────────────────────────────────────────────────────────────────────────────

res_plot = 0
# Sample rates to test (Hz)
sample_rates = sr0 * np.pow(2, np.arange(Nsrs))
sr_ref = sample_rates[-1]

print(f"Running reference simulation at sr = {sr_ref} Hz …")
fname_ref = path.join(result_folder, "testSingleReed_ref.hdf5")
ref = run_simulation(sr_ref, duration, fname_ref)
q_ref = ref["q"]                      # shape (N_ref, 3)
q_ref = q_ref[::int(sr_ref / sr0)]
if (idx_plot == Nsrs - 1):
    res_plot = ref

# ──────────────────────────────────────────────────────────────────────────────
# Convergence study
# ──────────────────────────────────────────────────────────────────────────────

l2_errors = []

for i, sr in enumerate(sample_rates[:-1]):
    print(f"  Running simulation at sr = {sr} Hz …")
    fname = path.join(result_folder, f"testSingleReed_{sr}.hdf5")
    res = run_simulation(sr, duration, fname)
    if (i == idx_plot):
        res_plot = res

    # Restrict to the analysis window on the reference grid
    idxmin_ref = int(tmin * sr0)
    idxmax_ref = int(tmax * sr0)
    idxmin_cur = int(tmin * sr)
    idxmax_cur = int(tmax * sr)

    q_cur_window = res["q"][idxmin_cur:idxmax_cur]
    q_ref_window = q_ref[idxmin_ref:idxmax_ref]

    # Interpolate current solution onto the reference grid (within window)
    q_cur_resampled = q_cur_window[::int(sr/sr0)]

    diff = q_cur_resampled - q_ref_window
    l2_errors.append(l2_norm(diff) / l2_norm(q_ref_window))

# ──────────────────────────────────────────────────────────────────────────────
# Convergence plot
# ──────────────────────────────────────────────────────────────────────────────

fig_conv, ax_conv = plt.subplots(
    figsize=set_size(width=fig_width, height_ratio=1))

ax_conv.loglog(sample_rates[:-1], l2_errors, "o-")

# Overlay reference slopes for visual guidance
sr_arr = np.array(sample_rates[:-1], dtype=float)
for order, ls in [(1, "--")]:
    ref_slope = l2_errors[0] * (sr_arr[0] / sr_arr) ** order
    ax_conv.loglog(sr_arr, ref_slope, color="grey",
                   ls=ls, lw=0.8, label=f"Order {order}")

ax_conv.set_xlabel("Sample rate (Hz)")
ax_conv.set_ylabel(r"$\| q - q_{\mathrm{ref}} \|_{L_2}$ (m)")
ax_conv.legend(frameon=True)
ax_conv.grid(which="both", ls=":")
plt.tight_layout()
fig_conv.savefig(
    path.join(figure_folder, "Single_Reed_convergence.png"),
    bbox_inches="tight",
)
print("Convergence plot saved.")

# ──────────────────────────────────────────────────────────────────────────────
# Displacements and flow plots
# ──────────────────────────────────────────────────────────────────────────────

sr = sample_rates[idx_plot]
time = res_plot["time"]
q = res_plot["q"]
layPosition = res_plot["layPosition"]
resonatorFlow = res_plot["resonatorFlow"]
meanFlow = res_plot["meanFlow"]
Pext = res_plot["Pext"]
PextSub = res_plot["PextSub"]
PextSup = res_plot["PextSup"]
Pdiss = res_plot["Pdiss"]
PdissFlow = res_plot["PdissFlow"]
PdissReed = res_plot["PdissReed"]
Pstored = res_plot["Pstored"]
PstoredKinetic = res_plot["PstoredKinetic"]
PstoredPotential = res_plot["PstoredPotential"]
radiatedPressure = res_plot["radiatedPressure"][::int(sr / sr0)]

write(path.join(result_folder, "clarinet.wav"), sr0,
      radiatedPressure / np.max(np.abs(radiatedPressure)))

idxmin = int(tmin * sr)
idxmax = int(tmax * sr)

# Powers
fig = plt.figure(figsize=set_size(width=fig_width, height_ratio=0.7))

linestyles = ["-", "--", "-."]

plt.plot(time[idxmin:idxmax], Pstored[idxmin:idxmax],
         label="Stored", color="blue", ls=linestyles[0])
# plt.plot(time[idxmin:idxmax], PstoredKinetic[idxmin:idxmax],
#          label="Stored, kinetic", color="blue", ls=linestyles[1])
# plt.plot(time[idxmin:idxmax], PstoredPotential[idxmin:idxmax],
#          label="Stored, potential", color="blue", ls=linestyles[2])

plt.plot(time[idxmin:idxmax], Pext[idxmin:idxmax],
         label="External", color="red", ls=linestyles[0])
# plt.plot(time[idxmin:idxmax], PextSub[idxmin:idxmax],
#          label="External, mouth", color="red", ls=linestyles[1])
# plt.plot(time[idxmin:idxmax], PextSup[idxmin:idxmax],
#          label="External, resonator", color="red", ls=linestyles[2])

plt.plot(time[idxmin:idxmax], Pdiss[idxmin:idxmax],
         label="Dissipated", color="green", ls=linestyles[0])
# plt.plot(time[idxmin:idxmax], PdissFlow[idxmin:idxmax],
#          label="Dissipated, flow jet", color="green", ls=linestyles[1])
# plt.plot(time[idxmin:idxmax], PdissReed[idxmin:idxmax],
#          label="Dissipated, mechanical", color="green", ls=linestyles[2])

plt.xlim([tmin, tmax])
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.legend(frameon=True, loc=1)
plt.grid()
plt.tight_layout()
fig.savefig(
    path.join(figure_folder, "Single_Reed_powers.png"),
    bbox_inches="tight",
)
print("\n")
print(f"Power balance mean relative error for fs={sr}:", "{0:0.2E}".format(l2_norm(
    Pstored + Pext + Pdiss) / l2_norm(Pdiss)))

# Masses displacements and glottal flow
fig2, axs = plt.subplots(
    figsize=set_size(width=fig_width, height_ratio=0.5, subplots=(2, 1)),
    nrows=2, ncols=1, sharex=True, height_ratios=[1, 1],
)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axs[0].plot(time[idxmin:idxmax],
            (layPosition - q[idxmin:idxmax]) * 1000, label="Lower", ls="--")
# axs[0].plot(time[idxmin:idxmax],
#             (layPosition - q_ref[idxmin:idxmax]) * 1000, label="Lower", ls="-.")
axlims = axs[0].get_ylim()
axs[0].axhspan(-1, 0, alpha=0.3, color="grey")
axs[0].set_ylim(axlims)

axs[1].plot(time, resonatorFlow, color="red", label="Total")
axs[1].plot(time, meanFlow, color="green", ls="--", label="Pressure-induced")
axs[1].plot(time, resonatorFlow - meanFlow,
            color="blue", ls="-.", label="Reed-induced")

plt.xlim([tmin, tmax])
plt.xlabel("Time (s)")
axs[0].set_ylabel("Opening\n(mm)",           multialignment="center")
axs[1].set_ylabel("Flow\n(m$^3.$s$^{-1}$)",  multialignment="center")
axs[1].legend(frameon=True, loc=1)
fig2.align_ylabels()

for ax in axs:
    ax.grid()

plt.tight_layout()
fig2.subplots_adjust(hspace=0.25)
fig2.savefig(
    path.join(figure_folder, "Single_Reed_displacement.png"),
    bbox_inches="tight",
)

plt.show()
