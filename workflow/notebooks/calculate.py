import time
from os import PathLike

import numpy as np
import pandas as pd
from coulson.interface import mol_from_xyz, process_rdkit_mol
from coulson.ppp import (
    PPPCalculator,
    calculate_dsp,
    calculate_exchange,
    homo_lumo_overlap,
)


def calculate_compound(filename: str | PathLike) -> dict[str, list[int]]:
    """Calculate excitation roperties of molecule."""
    # Process filename with RDKit mol detection from XYZ
    start_time = time.time()
    mol = mol_from_xyz(filename)
    input_data, mask = process_rdkit_mol(mol)
    if not any(mask) is True:
        raise ValueError("Molecule does not have pi system.")

    # Perform SCF calculation
    ppp = PPPCalculator(input_data)
    ppp.scf(max_iter=500)

    # Calcualte exchange integral
    exchange = calculate_exchange(ppp)

    # Calculate HOMO-LUMO overlap
    overlap = homo_lumo_overlap(ppp)

    # Calculate S1 energies and oscillator strengths
    ppp.ci(n_states=3)
    energy_s1_cis = ppp.ci_energies[0] - ppp.electronic_energy
    oscillator_strength = ppp.oscillator_strengths[0]

    # Calculate T1 energies
    ppp.ci(n_states=3, multiplicity="triplet")
    energy_t1_cis = ppp.ci_energies[0] - ppp.electronic_energy

    # Calculate DSP contribution
    dsp_scf, _ = calculate_dsp(ppp)
    dsp_cis, _ = calculate_dsp(
        ppp, ci=True, energy_s_1=energy_s1_cis, energy_t_1=energy_t1_cis
    )
    end_time = time.time()
    run_time = end_time - start_time

    results = {
        "s1_cis": [energy_s1_cis],
        "t1_cis": [energy_t1_cis],
        "exchange_integral": [exchange],
        "dsp_scf": [dsp_scf],
        "dsp_cis": [dsp_cis],
        "homo_lumo_overlap": [overlap],
        "oscillator_strength": [oscillator_strength],
        "run_time": run_time,
    }

    return results


def main():
    # Perform calculation
    try:
        results = calculate_compound(snakemake.input[0])
    except ValueError:
        results = {
            "s1_cis": [np.nan],
            "t1_cis": [np.nan],
            "exchange_integral": [np.nan],
            "dsp_scf": [np.nan],
            "dsp_cis": [np.nan],
            "homo_lumo_overlap": [np.nan],
            "oscillator_strength": [np.nan],
        }
    # Write output
    idx = snakemake.wildcards.sample
    data = {"id": [idx], **results}
    df = pd.DataFrame(data).set_index("id")
    df.to_csv(snakemake.output[0])


if __name__ == "__main__":
    main()
