from coulson.interface import process_rdkit_mol, mol_from_xyz
from coulson.ppp import (
    calculate_exchange,
    calculate_dsp,
    homo_lumo_overlap,
    PPPCalculator,
)
import pandas as pd

idx = snakemake.wildcards.sample

mol = mol_from_xyz(snakemake.input[0])
input_data, mask = process_rdkit_mol(mol)
ppp = PPPCalculator(input_data)
ppp.scf(max_iter=500)

# Calcualte exchange integral
exchange = calculate_exchange(ppp)

# Calculate HOMO-LUMO overlap
overlap = homo_lumo_overlap(ppp)

# Calculate S1 energies and oscillator strengths
ppp.ci(n_states=3)
energy_s1_cis = ppp.ci_energies[1] - ppp.ci_energies[0]

oscillator_strength = ppp.oscillator_strengths[0]

# Calculate T1 energies
ppp.ci(n_states=3, multiplicity="triplet")
energy_t1_cis = ppp.ci_energies[1] - ppp.ci_energies[0]

# Calculate DSP contribution
dsp_scf, excitations = calculate_dsp(ppp)
dsp_cis, excitations = calculate_dsp(
    ppp, ci=True, energy_s_1=energy_s1_cis, energy_t_1=energy_t1_cis
)

data = {
    "id": [idx],
    "s1_cis": [energy_s1_cis],
    "t1_cis": [energy_t1_cis],
    "exchange_integral": [exchange],
    "dsp_scf": [dsp_scf],
    "dsp_cis": [dsp_cis],
    "homo_lumo_overlap": [overlap],
    "oscillator_strength": [oscillator_strength],
}
df = pd.DataFrame(data).set_index("id")
df.to_csv(snakemake.output[0])
