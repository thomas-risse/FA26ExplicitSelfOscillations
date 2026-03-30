import json
import numpy as np
import matplotlib.pyplot as plt

import h5py
import subprocess
from os import path, makedirs
from helper_plots import set_size


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(sr: int, duration: float, fname: str) -> dict:
    """Write inputs, run the C++ solver, and return the relevant arrays."""
    t = np.linspace(0, duration, int(sr * duration))

    # Excitation
    Pin0 = 800
    trise = 0.001
    Pin = np.ones_like(t) * Pin0
    Pin[: int(trise * sr)] = np.linspace(0, Pin0, int(trise * sr))

    with h5py.File(fname, "w") as f:
        f.attrs["sr"] = sr
        f.attrs["duration"] = duration
        f["Pmouth"] = Pin

    subprocess.run(
        ["../build/RunLarynx", f"{path.realpath(fname)}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with h5py.File(fname, "r") as f:
        return {
            "q":               f["foldDisplacement"][:][::2],       # (N, 3)
            "restPositions":   f["restPositions"][:],
            "supGlottalFlow":  f["supGlottalFlow"][:],
            "meanGlottalFlow": f["meanGlottalFlow"][:],
            "pressureDrop":    f["pressureDrop"][:],
            "time":            f["time"][:],
            "Pext":            f["Pext"][:],
            "PextSub":         f["PextSub"][:],
            "PextSup":         f["PextSup"][:],
            "Pdiss":           f["Pdiss"][:],
            "PdissFlow":       f["PdissFlow"][:],
            "PdissFolds":      f["PdissFolds"][:],
            "Pstored":         f["Pstored"][:],
        }


def downsample(signal: np.ndarray, factor) -> np.ndarray:
    """
    Down-sample `signal` (recorded at `sr`) onto the time grid of `sr_ref`
    by simple nearest-neighbour index mapping.
    Works correctly only when sr >= sr_ref (i.e. sr_ref is the coarsest grid).
    For the general case we use linear interpolation on the time axis.
    """
    return signal[::factor]


def l2_norm(a: np.ndarray) -> float:
    """Discrete L2 norm (RMS-like, normalised by length)."""
    return np.sqrt(np.mean(a ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Solver parameters
# ──────────────────────────────────────────────────────────────────────────────

if not path.exists("results"):
    makedirs("results")

duration = 0.1

# ──────────────────────────────────────────────────────────────────────────────
# Reference simulation (highest sample rate)
# ──────────────────────────────────────────────────────────────────────────────

# Sample rates to test (Hz) – must include the reference (last / highest)
sample_rates = 44100 * np.pow(2, np.arange(8))
sr0 = sample_rates[0]
sr_ref = sample_rates[-1]

print(f"Running reference simulation at sr = {sr_ref} Hz …")
fname_ref = "results/testVoice_ref.hdf5"
ref = run_simulation(sr_ref, duration, fname_ref)
q_ref = ref["q"]                      # shape (N_ref, 3)
q_ref = downsample(q_ref, int(sr_ref / sr0))
print(q_ref)
# ──────────────────────────────────────────────────────────────────────────────
# Convergence study
# ──────────────────────────────────────────────────────────────────────────────

# We study convergence only over the steady-state window used for the plots
tmin, tmax = 0.03, 0.05

l2_errors = []

for sr in sample_rates[:-1]:
    print(f"  Running simulation at sr = {sr} Hz …")
    fname = f"results/testVoice_{sr}.hdf5"
    res = run_simulation(sr, duration, fname)

    # Restrict to the analysis window on the reference grid
    idxmin_ref = int(tmin * sr0)
    idxmax_ref = int(tmax * sr0)
    idxmin_cur = int(tmin * sr)
    idxmax_cur = int(tmax * sr)

    q_cur_window = res["q"][idxmin_cur:idxmax_cur]
    q_ref_window = q_ref[idxmin_ref:idxmax_ref]

    # Interpolate current solution onto the reference grid (within window)
    q_cur_resampled = downsample(q_cur_window, int(sr/sr0))

    diff = q_cur_resampled - q_ref_window
    print(q_cur_resampled.shape)
    l2_errors.append(l2_norm(diff))

sr_tested = sample_rates[:-1]

# ──────────────────────────────────────────────────────────────────────────────
# Convergence plot
# ──────────────────────────────────────────────────────────────────────────────

fig_conv, ax_conv = plt.subplots(
    figsize=set_size(width='FA', height_ratio=0.6))

ax_conv.loglog(sr_tested, l2_errors, "o-")
print(l2_errors)

# Overlay reference slopes for visual guidance
sr_arr = np.array(sr_tested, dtype=float)
for order, ls in [(1, "--"), (2, ":")]:
    ref_slope = l2_errors[0] * (sr_arr[0] / sr_arr) ** order
    ax_conv.loglog(sr_arr, ref_slope, color="grey",
                   ls=ls, lw=0.8, label=f"Order {order}")

ax_conv.set_xlabel("Sample rate (Hz)")
ax_conv.set_ylabel(r"$\| q - q_{\mathrm{ref}} \|_{L_2}$ (m)")
ax_conv.set_title("Convergence of vocal-fold displacement w.r.t. sample rate")
ax_conv.legend(frameon=True)
ax_conv.grid(which="both", ls=":")
plt.tight_layout()
fig_conv.savefig(
    "/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_convergence.pdf",
    bbox_inches="tight",
)
print("Convergence plot saved.")

# ──────────────────────────────────────────────────────────────────────────────
# Original plots (using the reference / highest-sr run)
# ──────────────────────────────────────────────────────────────────────────────

sr = sr_ref
time = ref["time"]
q = ref["q"]
restPositions = ref["restPositions"]
supGlottalFlow = ref["supGlottalFlow"]
meanGlottalFlow = ref["meanGlottalFlow"]
pressureDrop = ref["pressureDrop"]
Pext = ref["Pext"]
PextSub = ref["PextSub"]
PextSup = ref["PextSup"]
Pdiss = ref["Pdiss"]
PdissFlow = ref["PdissFlow"]
PdissFolds = ref["PdissFolds"]
Pstored = ref["Pstored"]

idxmin = int(tmin * sr)
idxmax = int(tmax * sr)

# Powers
fig = plt.figure(figsize=set_size(width='FA', height_ratio=0.7))

plt.plot(time[idxmin:idxmax], Pstored[idxmin:idxmax], label="Stored")
plt.plot(time[idxmin:idxmax], Pext[idxmin:idxmax],    label="External")
plt.plot(time[idxmin:idxmax], Pdiss[idxmin:idxmax],   label="Dissipated")

plt.xlim([tmin, tmax])
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.legend(frameon=True, loc=1)
plt.grid()
plt.tight_layout()
fig.savefig(
    "/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_powers.pdf",
    bbox_inches="tight",
)

# Masses displacements and glottal flow
fig2, axs = plt.subplots(
    figsize=set_size(width='FA', height_ratio=0.5, subplots=(2, 1)),
    nrows=2, ncols=1, sharex=True, height_ratios=[1, 0.5],
)

plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axs[0].plot(time[idxmin:idxmax],
            (restPositions[0] - q[idxmin:idxmax, 0]) * 1000, label="Lower", ls="--")
axs[0].plot(time[idxmin:idxmax],
            (restPositions[1] - q[idxmin:idxmax, 1]) * 1000, label="Upper", ls="-")
axs[0].plot(time[idxmin:idxmax],
            (restPositions[2] - q[idxmin:idxmax, 2]) * 1000, label="Body",  ls=":")
axlims = axs[0].get_ylim()
axs[0].axhspan(-1, 0, alpha=0.3, color="grey")
axs[0].set_ylim(axlims)

axs[1].plot(time, supGlottalFlow, color="red")

plt.xlim([tmin, tmax])
plt.xlabel("Time (s)")
axs[0].set_ylabel("Opening\n(mm)",            multialignment="center")
axs[1].set_ylabel("Flow\n(m$^3.$s$^{-1}$)",  multialignment="center")
axs[0].legend(frameon=True, loc=1)
fig2.align_ylabels()

for ax in axs:
    ax.grid()

plt.tight_layout()
fig2.subplots_adjust(hspace=0.25)
fig2.savefig(
    "/Users/risse/Projects/NonlinearDissipations/FA2026/assets/Voice_displacements.pdf",
    bbox_inches="tight",
)

plt.show()
