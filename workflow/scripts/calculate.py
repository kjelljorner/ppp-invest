import coulson

from coulson.interface import process_coordinates
from coulson.ppp import calculate_exchange, calculate_dsp, homo_lumo_overlap, PPPCalculator
from morfeus.io import read_xyz
import pandas as pd

elements, coordinates = read_xyz(snakemake.input[0])
input_data, mask = process_coordinates(elements, coordinates)
ppp = PPPCalculator(input_data)
ppp.scf()

# Calcualte exchange integral
exchange = calculate_exchange(ppp)

# Calculate DSP contribution
dsp = calculate_dsp(ppp)

# Calculate HOMO-LUMO overlap
overlap = homo_lumo_overlap(ppp)

# Calculate S1 energies and oscillator strengths
ppp.ci(n_states=3)
energy_s1_cis = (ppp.ci_energies[1] - ppp.ci_energies[0])
    
oscillator_strength = ppp.oscillator_strengths[0]

# Calculate T1 energies
ppp.ci(n_states=3, multiplicity="triplet")
energy_t1_cis = (ppp.ci_energies[1] - ppp.ci_energies[0])

data = {
    "id": [snakemake.wildcards.sample],
    "s1_cis": [energy_s1_cis],
    "t1_cis": [energy_t1_cis],
    "exchange_integral": [exchange],
    "dsp_correction": [dsp],
    "homo_lumo_overlap": [overlap],
    "oscillator_strength": [oscillator_strength]
    } 
df = pd.DataFrame(data).set_index("id")
df.to_csv(snakemake.output[0])
