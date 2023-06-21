import coulson

from coulson.interface import process_coordinates, get_pyscf_mf
from coulson.ppp import calculate_exchange, calculate_dsp, homo_lumo_overlap, PPPCalculator
from morfeus.io import read_xyz
import pandas as pd
from pyscf.cc import CCSD
from pyscf.adc import ADC
from pyscf.tdscf import TDA
from pathlib import Path

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


# Calculate S1 and T1 energies with ADC(3)

# Calculate S1 and T1 with TDA
mf = get_pyscf_mf(ppp, spin=0)
mf.verbose = 0
mf.scf()
tda = TDA(mf)
energy_s1_tda = tda.kernel(nstates=1)[0][0]
tda.singlet = False
energy_t1_tda = tda.kernel(nstates=1)[0][0]

# Calculate S1 and T1 energies with EOM-CCSD
#mf = get_pyscf_mf(ppp, spin=0)
#mf.verbose = 0
#mf.scf()
#mf.kernel()
ccsd = CCSD(mf)
ccsd.verbose = 0
ccsd.ccsd()
#results = ccsd.kernel()
#e_corr, t_1, t_2 = ccsd.ccsd()
#eris = ccsd.ao2mo()
#energy_s1_eom = ccsd.eomee_ccsd_singlet(nroots=1)[0]
energy_s1_eom = ccsd.eeccsd(nroots=5)[0]
#energy_s1_eom, _ = ccsd.eomee_ccsd_singlet(eris=eris, nroots=1)
energy_t1_eom = ccsd.eomee_ccsd_triplet(nroots=1)[0]
#energy_t1_eom, _ = ccsd.eomee_ccsd_triplet(eris=eris, nroots=1)
data = {
    "id": [snakemake.wildcards.sample],
    "s1_cis": [energy_s1_cis],
    "s1_tda": [energy_s1_tda],
    "s1_eom": [energy_s1_eom],
    "t1_cis": [energy_t1_cis],
    "t1_tda": [energy_t1_tda],
    "t1_eom": [energy_t1_eom],    
    "exchange_integral": [exchange],
    "dsp_correction": [dsp],
    "homo_lumo_overlap": [overlap],
    "oscillator_strength": [oscillator_strength]
    } 
df = pd.DataFrame(data).set_index("id")
df.to_csv(snakemake.output[0])
