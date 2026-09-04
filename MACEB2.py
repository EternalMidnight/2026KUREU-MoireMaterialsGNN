"""Imports and initializations""" #These are all of the imports and there may be overlaps or unnecessary ones but it runs fine.
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
import random 
import shutil

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
    force_field: pd.DataFrame #This is the datatype used to get our needed data from OUTCAR and use on MACE.
    directory: str | None  #This for tagging data with where they came from (different folders), assume user puts different structure types in different folders.
    #Being honest I am not exactly sure how to trigger this feature for directory, I just used one folder. Grant's project may have needed it.

"""Splitting the files"""

source_dir = "./OUTCAR"
training_dir = "./OUTCAR_training"
testing_dir = "./OUTCAR_testing"

# Create destination directories
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# Find all OUTCAR files
files = []

for root, dirs, filenames in os.walk(source_dir):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        files.append(filepath)

t = int(time.time())
random.seed(t)
random.shuffle(files)

# 80/20 split
split = round(len(files) * 0.80)

training_files = files[:split]
testing_files = files[split:]

# Copy files
for filepath in training_files:
    shutil.copy2(
        filepath,
        os.path.join(training_dir, os.path.basename(filepath))
    )

for filepath in testing_files:
    shutil.copy2(
        filepath,
        os.path.join(testing_dir, os.path.basename(filepath))
    )

print(f"Total files: {len(files)}")
print(f"Training files: {len(training_files)}")
print(f"Testing files: {len(testing_files)}")



"""Getting the data from training set"""

# Stores all parsed ionic steps from every OUTCAR file.
# Each entry corresponds to one OUTCAR and contains a list of ModelDataType objects.
training_dataset_iterations: list[list[ModelDataType]] = []

# Stores the corresponding atomic symbols for every ionic step.
# Structure:
# dataset_symbols[file][ionic_step][atom]
training_dataset_symbols: list[list[list[str]]] = []

# Recursively search through the OUTCAR directory, taking the data from the OUTCAR files and storing it.
for root, dirs, files in os.walk(training_dir):
    # Process every file found
    for filename in files:
        with io.open(os.path.join(root, filename), buffering = 5) as file:
            directory = os.path.basename(os.path.dirname(root))
            data = file.read()
            match = re.findall(
                # in order of matches: | major iteration | minor iteration | table of position vectors and force vectors | total energy [eV] |
                r"(?s)Iteration *(\d+)\( *(\d+).*?direct lattice vectors.*?\n(.*?)\n\n.*?TOTAL-FORCE.*?(?:.*?\n){2}(.+?) -----.*?FREE ENERGIE OF THE ION-ELECTRON SYSTEM.*?energy\(sigma->0\)\s*=\s*([-\d.]+)", 
                data
            ) #IMPORTANT!!! This is the regex that gets the right data from the OUTCAR files. Double check what you want to get and what you do get.

            match_symbols = re.findall(r"VRHFIN\s*=([a-zA-z]*)", data)  # Read element names from the POTCAR information
            match_counts = re.findall(r"ions per type\s*=\s*((?:\d+\s*)+)", data)  # Read the number of atoms of each element

            # Skip files that failed to parse
            if len(match_counts) == 0 or len(match) == 0:
                continue

            elements: list[str] = match_symbols
            counts: list[int] = list(map(int, match_counts[0].split()))


            # Convert lattice-vector text into a DataFrame
            def lattice(m):
                # Fix concatenated numbers like '0.000000000-18.016899180' → '0.000000000 -18.016899180'
                # This regex looks for a number (int or float) followed by a minus sign and another number without space
                m_fixed = re.sub(r'(?<=[\d.])-(?=\d)', ' -', m)

                df = pd.read_csv(StringIO(m_fixed), sep=r'\s+', header=None, names=["a", "b", "c", "aa", "bb", "cc"])
                return df.loc[:, ["a", "b", "c"]]    # Step 1: Add space between negative numbers that follow digits (e.g. 1.23-4.56 → 1.23 -4.56)

            # Convert the position/force table into a DataFrame
            def forces(m):
                return pd.read_csv(StringIO(m), sep=r'\s+', header=None, names=["x", "y", "z", "Fx", "Fy", "Fz"])

            iterations: list[ModelDataType] = [ModelDataType(m[0], m[1], m[4], lattice(m[2]), forces(m[3]), directory)
                                               for m in match] # Create one ModelDataType object for every ionic step
            symbols: list[list[str]] = [[elm for elm, count in zip(elements, counts) for _ in range(count)] for _ in range(len(iterations))]

            training_dataset_iterations.append(iterations) # Store the parsed data for this OUTCAR
            #print(iterations)
            training_dataset_symbols.append(symbols) # Store the parsed data for this OUTCAR

"""Reformatting and preparing the training data"""

training_dataset_atoms: list[list[Atoms]] = [] #This huge chunk effectively tries to store all the data gathered above as an ase atom datatype, following ase format.
for dataset, symbols in zip(training_dataset_iterations, training_dataset_symbols):
    try:
        positions_dataset: list[pd.DataFrame] = [d.force_field.loc[:, ["x","y","z"]].to_numpy() for d in dataset]
        forces_dataset: list[pd.DataFrame] = [d.force_field.loc[:, ["Fx", "Fy", "Fz"]].to_numpy() for d in dataset]
        energies_dataset: list[float] = [d.total_energy_ev for d in dataset]
        lattice_dataset: list[float] = [d.lattice_vectors.to_numpy() for d in dataset]
        directories_dataset: str = [d.directory for d in dataset][0]

        molecular_iterations: list[Atoms] = [Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True) for symbols, positions, cell in zip(symbols, positions_dataset, lattice_dataset)] # merge dataset into list[Atoms]

        for iteration, force, energy in zip(molecular_iterations, forces_dataset, energies_dataset):
            iteration.arrays['REF_forces'] = force
            iteration.info['REF_energy'] = energy
            iteration.info['head'] = "zhoulab"
            iteration.info['force_magnitude'] = np.sqrt(np.mean(np.sum(force**2, axis=1)))
            iteration.info['data_dir'] = directories_dataset

        training_dataset_atoms.append(molecular_iterations)
    except Exception as e:
        print("EXCEPTION!")
        print(e)
        print(repr(e)) #If there is exception, make sure to notice it and provide exactly what is the error/exception.
        continue

training_merged = list(itertools.chain(*training_dataset_atoms)) 
#This line flattens dataset_atoms into one list, as dataset_atoms is actually a 2D list. Structures from the same OUTCAR but different ionic steps are in lists, and a 
#list of these lists is dataset_atoms.

# Filter to only complete atoms
training_validated_merged = [atoms for atoms in training_merged
                   if 'REF_forces' in atoms.arrays
                    and 'REF_energy' in atoms.info
                    and 'numbers' in atoms.arrays
                    and 'positions' in atoms.arrays
                    and set(atoms.get_chemical_symbols()).issubset({'N', 'B'}) 
                    # and {atoms.info["data_dir"]}.issubset({'bilayer'})
                    #The above lines checks for data completeness and validity.


                    and atoms.info['force_magnitude'] < 0.8  # Exclude non-physical forces with a hard cutoff
                    and np.abs(np.float64(atoms.info['REF_energy']))/len(atoms.get_chemical_symbols()) < 10
                    and np.float64(atoms.info['REF_energy']) <= 0 #
                    #The above lines checks for Physics correctness using Physics theory.
                   ]

print(f"Training Original count: {len(training_merged)}")
print(f"Training Complete count: {len(training_validated_merged)}")

"""Getting the data from testing set"""

# Stores all parsed ionic steps from every OUTCAR file.
# Each entry corresponds to one OUTCAR and contains a list of ModelDataType objects.
testing_dataset_iterations: list[list[ModelDataType]] = []

# Stores the corresponding atomic symbols for every ionic step.
# Structure:
# dataset_symbols[file][ionic_step][atom]
testing_dataset_symbols: list[list[list[str]]] = []

# Recursively search through the OUTCAR directory, taking the data from the OUTCAR files and storing it.
for root, dirs, files in os.walk(testing_dir):
    # Process every file found
    for filename in files:
        with io.open(os.path.join(root, filename), buffering = 5) as file:
            directory = os.path.basename(os.path.dirname(root))
            data = file.read()
            match = re.findall(
                # in order of matches: | major iteration | minor iteration | table of position vectors and force vectors | total energy [eV] |
                r"(?s)Iteration *(\d+)\( *(\d+).*?direct lattice vectors.*?\n(.*?)\n\n.*?TOTAL-FORCE.*?(?:.*?\n){2}(.+?) -----.*?FREE ENERGIE OF THE ION-ELECTRON SYSTEM.*?energy\(sigma->0\)\s*=\s*([-\d.]+)", 
                data
            ) #IMPORTANT!!! This is the regex that gets the right data from the OUTCAR files. Double check what you want to get and what you do get.

            match_symbols = re.findall(r"VRHFIN\s*=([a-zA-z]*)", data)  # Read element names from the POTCAR information
            match_counts = re.findall(r"ions per type\s*=\s*((?:\d+\s*)+)", data)  # Read the number of atoms of each element

            # Skip files that failed to parse
            if len(match_counts) == 0 or len(match) == 0:
                continue

            elements: list[str] = match_symbols
            counts: list[int] = list(map(int, match_counts[0].split()))


            # Convert lattice-vector text into a DataFrame
            def lattice(m):
                # Fix concatenated numbers like '0.000000000-18.016899180' → '0.000000000 -18.016899180'
                # This regex looks for a number (int or float) followed by a minus sign and another number without space
                m_fixed = re.sub(r'(?<=[\d.])-(?=\d)', ' -', m)

                df = pd.read_csv(StringIO(m_fixed), sep=r'\s+', header=None, names=["a", "b", "c", "aa", "bb", "cc"])
                return df.loc[:, ["a", "b", "c"]]    # Step 1: Add space between negative numbers that follow digits (e.g. 1.23-4.56 → 1.23 -4.56)

            # Convert the position/force table into a DataFrame
            def forces(m):
                return pd.read_csv(StringIO(m), sep=r'\s+', header=None, names=["x", "y", "z", "Fx", "Fy", "Fz"])

            iterations: list[ModelDataType] = [ModelDataType(m[0], m[1], m[4], lattice(m[2]), forces(m[3]), directory)
                                               for m in match] # Create one ModelDataType object for every ionic step
            symbols: list[list[str]] = [[elm for elm, count in zip(elements, counts) for _ in range(count)] for _ in range(len(iterations))]

            testing_dataset_iterations.append(iterations) # Store the parsed data for this OUTCAR
            #print(iterations)
            testing_dataset_symbols.append(symbols) # Store the parsed data for this OUTCAR

"""Reformatting and preparing the testing data"""

testing_dataset_atoms: list[list[Atoms]] = [] #This huge chunk effectively tries to store all the data gathered above as an ase atom datatype, following ase format.
for dataset, symbols in zip(testing_dataset_iterations, testing_dataset_symbols):
    try:
        positions_dataset: list[pd.DataFrame] = [d.force_field.loc[:, ["x","y","z"]].to_numpy() for d in dataset]
        forces_dataset: list[pd.DataFrame] = [d.force_field.loc[:, ["Fx", "Fy", "Fz"]].to_numpy() for d in dataset]
        energies_dataset: list[float] = [d.total_energy_ev for d in dataset]
        lattice_dataset: list[float] = [d.lattice_vectors.to_numpy() for d in dataset]
        directories_dataset: str = [d.directory for d in dataset][0]

        molecular_iterations: list[Atoms] = [Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True) for symbols, positions, cell in zip(symbols, positions_dataset, lattice_dataset)] # merge dataset into list[Atoms]

        for iteration, force, energy in zip(molecular_iterations, forces_dataset, energies_dataset):
            iteration.arrays['REF_forces'] = force
            iteration.info['REF_energy'] = energy
            iteration.info['head'] = "zhoulab"
            iteration.info['force_magnitude'] = np.sqrt(np.mean(np.sum(force**2, axis=1)))
            iteration.info['data_dir'] = directories_dataset

        testing_dataset_atoms.append(molecular_iterations)
    except Exception as e:
        print("EXCEPTION!")
        print(e)
        print(repr(e)) #If there is exception, make sure to notice it and provide exactly what is the error/exception.
        continue

testing_merged = list(itertools.chain(*testing_dataset_atoms)) 
#This line flattens dataset_atoms into one list, as dataset_atoms is actually a 2D list. Structures from the same OUTCAR but different ionic steps are in lists, and a 
#list of these lists is dataset_atoms.

# Filter to only complete atoms
testing_validated_merged = [atoms for atoms in testing_merged
                   if 'REF_forces' in atoms.arrays
                    and 'REF_energy' in atoms.info
                    and 'numbers' in atoms.arrays
                    and 'positions' in atoms.arrays
                    and set(atoms.get_chemical_symbols()).issubset({'N', 'B'}) 
                    # and {atoms.info["data_dir"]}.issubset({'bilayer'})
                    #The above lines checks for data completeness and validity.


                    and atoms.info['force_magnitude'] < 0.8  # Exclude non-physical forces with a hard cutoff
                    and np.abs(np.float64(atoms.info['REF_energy']))/len(atoms.get_chemical_symbols()) < 10
                    and np.float64(atoms.info['REF_energy']) <= 0 #
                    #The above lines checks for Physics correctness using Physics theory.
                   ]


print(f"Testing Original count: {len(testing_merged)}")
print(f"Testing Complete count: {len(testing_validated_merged)}")


"""Outputting training data"""
write("./result/training-data-mlblbu-20.xyz", training_validated_merged, format="extxyz")
write("./result/testing-data-mlblbu-20.xyz", testing_validated_merged, format="extxyz")


"""Training"""

print("Python version:")
subprocess.run(["python", "--version"], check=True)
print("MACE version:", mace.__version__)

print("Starting MACE training at", datetime.datetime.now()) #Keeps track of training start time.

subprocess.run(
    [
        "mace_run_train",
        "--config", "./result/example_config.yaml",
    ],
    check=True,
) #Runs the MACE training process using subprocess in python (effectively running a bash script in python), using our yaml file.

print("Training completed at", datetime.datetime.now()) #Keeps track of training end time.

"""Test Model"""

#remove checkpoints since they may cause errors on retraining a model with the same name but a different architecture
for file in glob.glob("./*.pt"):
    os.remove(file)


warnings.filterwarnings("ignore")

#IMPORTANT!!! I did not read through Grant's evaluation code line by line, I think it works fine, but I won't be able to comment on it.
def eval_mace(configs, model, output, device="cpu", dtype="float64"): #Here is all of the settings for the model testing, the output of this is an .xyz file.
    os.makedirs(os.path.dirname(output), exist_ok=True)
    sys.argv = ["program", "--configs", configs, "--model", model, "--output", output,
            "--device", device, "--default_dtype", dtype] 

    mace_eval_configs_main() #Seem to just call the MACE model testing code from MACE.

def plot_mace_results(test_data_path: str, title: str): #Here is all of the formatting for the output visualizations.
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
    axes[0].set_ylim(y_lo-.01, y_hi+.01) #Here is where you set the boundaries for the Energy plot, if you want to zoom in or zoom out, change the numbers here.
    #Currently I set it at 0.01 and feels its mostly appropriate, making these values even smaller to zoom in, and the opposite to zoom out.

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
plt.close(fig3) #Should be self explanatory, take the right files, create the images, and then save it to your given file name.