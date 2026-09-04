"""Imports and initializations"""
import os
import subprocess
import sys

# Standard library
import datetime
import glob
import io
import itertools
import re
import time
import warnings
from dataclasses import dataclass
from io import StringIO
from random import seed, shuffle

# Third-party
import matplotlib.pyplot as plt
import mace
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read, write
from mace.cli.eval_configs import main as mace_eval_configs_main
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

@dataclass
class ModelDataType:
    major_iteration: int
    minor_iteration: int
    total_energy_ev: float
    lattice_vectors: pd.DataFrame
    force_field: pd.DataFrame
    directory: str | None  # mono, bi, or bulk if you named this right!

#remove checkpoints since they may cause errors on retraining a model with the same name but a different architecture


for file in glob.glob("./*.pt"):
    os.remove(file)

warnings.filterwarnings("ignore")

def eval_mace(configs, model, output, device="cpu", dtype="float64"):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    sys.argv = ["program", "--configs", configs, "--model", model, "--output", output,
            "--device", device, "--default_dtype", dtype]

    mace_eval_configs_main()

def plot_mace_results(test_data_path: str, title: str):
    test_data = read(test_data_path, index=":")

    test_energy_ref = np.array([atoms.info['REF_energy'] / len(atoms) for atoms in test_data])
    test_energy_pred = np.array([atoms.info['MACE_energy'] / len(atoms) for atoms in test_data])

    # Flatten forces for all configs
    test_forces_ref = np.concatenate([atoms.arrays['REF_forces'].flatten() for atoms in test_data])
    test_forces_pred = np.concatenate([atoms.arrays['MACE_forces'].flatten() for atoms in test_data])

    # Error metrics
    energy_rmse = np.sqrt(np.mean((test_energy_ref - test_energy_pred)**2))
    forces_rmse = np.sqrt(np.mean((test_forces_ref - test_forces_pred)**2))
    energy_corrcoef = np.corrcoef(test_energy_ref, test_energy_pred)[0, 1]
    forces_corrcoef = np.corrcoef(test_forces_ref, test_forces_pred)[0, 1]
    energy_mae = np.mean(np.abs(test_energy_ref - test_energy_pred))
    forces_mae = np.mean(np.abs(test_forces_ref - test_forces_pred))

    # Prepare colors by data_dir
    # Each atoms object gets a label from its info dict
    dirs = [atoms.info["data_dir"] for atoms in test_data]
    unique_dirs = sorted(set(dirs))
    cmap = get_cmap("gnuplot", len(unique_dirs))
    dir_to_color = {d: cmap(i) for i, d in enumerate(unique_dirs)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Energy parity plot ---
    energy_colors = [dir_to_color[d] for d in dirs]
    axes[0].scatter(test_energy_ref, test_energy_pred, c=energy_colors, alpha=0.6)
    e_min, e_max = min(test_energy_ref.min(), test_energy_pred.min()), max(test_energy_ref.max(), test_energy_pred.max())
    axes[0].plot([e_min, e_max], [e_min, e_max], 'r--')
    axes[0].set_xlabel('Reference Energy / Atom (eV)', fontsize=12)
    axes[0].set_ylabel('MACE Energy / Atom (eV)', fontsize=12)
    axes[0].set_title('Energy Parity (Test)', fontsize=12)
    x_lo, x_hi = np.percentile(test_energy_ref, [5, 99])
    y_lo, y_hi = np.percentile(test_energy_pred, [5, 99])
    axes[0].set_xlim(x_lo-.01, x_hi+.01)
    axes[0].set_ylim(y_lo-.01, y_hi+.01)

    # Build legend handles from unique data_dir
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=dir_to_color[d], markersize=8, label=d)
               for d in unique_dirs]
    axes[0].legend(handles=handles, title="data_dir", fontsize=9)

    # --- Force parity plot ---
    n_sample = min(5000, len(test_forces_ref))
    idx = np.random.choice(len(test_forces_ref), n_sample, replace=False)

    # Each force sample inherits the color of its parent structure
    # Build a list mapping each force to its parent's color
    force_colors = []
    for atoms in test_data:
        c = dir_to_color[atoms.info["data_dir"]]
        nforces = atoms.arrays['REF_forces'].size
        force_colors.extend([c] * nforces)
    force_colors = np.array(force_colors)[idx]

    axes[1].scatter(test_forces_ref[idx], test_forces_pred[idx], c=force_colors, alpha=0.6)
    f_min, f_max = min(test_forces_ref.min(), test_forces_pred.min()), max(test_forces_ref.max(), test_forces_pred.max())
    axes[1].plot([f_min, f_max], [f_min, f_max], 'r--', alpha=0.5)
    axes[1].set_xlabel('Reference Forces (eV/Å)', fontsize=12)
    axes[1].set_ylabel('MACE Forces (eV/Å)', fontsize=12)
    axes[1].set_title('Force Parity (Test)', fontsize=12)

    # --- Error Annotations ---
    xalign = 0.8
    axes[0].annotate(f"RMSE: {energy_rmse:.3f}\nCorr: {energy_corrcoef:.3f}\nMAE: {energy_mae:.3f}",
                     xy=(xalign, 0.1), xycoords='axes fraction', ha="center")
    axes[1].annotate(f"RMSE: {forces_rmse:.3f}\nCorr: {forces_corrcoef:.3f}\nMAE: {forces_mae:.3f}",
                     xy=(xalign, 0.1), xycoords='axes fraction', ha="center")

    # --- Outlier Annotations: Energy ---

    points_labels_energy = [(x, y, atoms) for x, y, atoms in zip(np.array([atoms.info['REF_energy'] / len(atoms) for atoms in test_data]),
                                           np.array([atoms.info['MACE_energy'] / len(atoms) for atoms in test_data]),
                                           # [atoms.get_chemical_formula() for atoms in test_data],
                                            test_data
                                            )
                     if abs(x-y) > 10000
             ]


  #  for x, y, atoms in points_labels_energy:
      #  axes[0].annotate(
       #     f"{atoms.get_chemical_formula()}, \nItr {atoms.info["iteration_info"]}",              # text label
         #   (x, y),             # point (x, y)
          #  textcoords="offset points",
          #  xytext=(5, 5),      # offset label slightly from point
          #  ha='left',
           # fontsize=8,
          #  color='black'
        #)


    # --- Rendering ---

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


# Evaluate fine-tuned model
eval_mace(
    configs="./result/testing-data-mlblbu-20.xyz",
    model="./MatPES_PBE_Final_Finetuned_compiled.model",
    output="./result/output1.xyz"
)
eval_mace(
    configs="./result/testing-data-mlblbu-20.xyz",
    model="./MatPES_PBE_Final_Finetuned_stagetwo_compiled.model",
    output="./result/output2.xyz"
)

# Evaluate foundation model
eval_mace(
    configs="./result/testing-data-mlblbu-20.xyz",
    model="./result/MACE-matpes-pbe-omat-ft.model",
    output="./result/f_output.xyz"
)

"""Plot results and output/visualize"""

fig1 = plot_mace_results("./result/output1.xyz", title="Fine Tuned S1")
fig1.savefig("fine_tuned1.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

fig2 = plot_mace_results("./result/f_output.xyz", title="Matpes PBE (Untuned)")
fig2.savefig("untuned.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

fig3 = plot_mace_results("./result/output2.xyz", title="Fine Tuned S2")
fig3.savefig("fine_tuned2.png", dpi=300, bbox_inches="tight")
plt.close(fig3)