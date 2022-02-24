import os
import sys

ROOT_DIR = os.getcwd()[:os.getcwd().rfind('quantum_HEOM')] + 'quantum_HEOM'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
from scipy import constants as c
from matplotlib import pyplot as plt
from quantum_heom import figures as figs
from quantum_heom import metadata as meta
from quantum_heom import utilities as util
from quantum_heom.quantum_system import QuantumSystem
from quantum_heom import bath
from quantum_heom import evolution as evo
from quantum_heom.lindbladian import LINDBLAD_MODELS

TRACE_MEASURES = ['squared', 'distance']
LEGEND_LABELS = {'local dephasing lindblad': 'Loc. Deph.',
                 'global thermalising lindblad': 'Glob. Therm.',
                 'local thermalising lindblad': 'Loc. Therm.',
                 'HEOM': 'HEOM',
                 'spin-boson': 'Spin-Boson',
                 'ohmic': 'Ohmic',
                 'debye': 'Debye',
                }
PLOT_TYPES = ['dynamics', 'spectral_density', 'compare_tr_dist',
              'fit_expo_tr_dist', 'integ_tr_dist_fxn_var', 'publication', 'ipr']


def plot_dynamics(systems, elements: [list, str] = None,
                  coherences: str = 'imag', trace_measure: list = None,
                  asymptote: bool = False, view_3d: bool = False,
                  save: bool = False):

    if not isinstance(systems, list):
        systems = [systems]
    assert systems, 'Must pass a QuantumSystem to plot dynamics for.'
    # Check the sites, timesteps, and time_intervals are the same for all
    # systems passed
    if len(systems) > 1:
        site_check = [sys.sites for sys in systems]
        timestep_check = [sys.timesteps for sys in systems]
        interval_check = [sys.time_interval for sys in systems]
        for var, name in [(site_check, 'sites'),
                          (timestep_check, 'timesteps'),
                          (interval_check, 'time_interval')]:
            assert var.count(var[0]) == len(var), ('For all systems passed the '
                                                   + name + ' must be'
                                                   ' the same.')
    sites = systems[0].sites
    # Checks the elements input, and convert to i.e. ['11', '21', ...] format
    elements = util.elements_from_str(sites, elements)
    if isinstance(coherences, str):
        assert coherences in ['real', 'imag'], ('Must pass coherences as either'
                                                ' "real" or "imag", or a list'
                                                ' containing both.')
        coherences = [coherences]
    elif isinstance(coherences, list):
        assert all(item in ['real', 'imag'] for item in coherences)
    else:
        raise ValueError('Invalid type for passing coherences')
    # Check trace_measure
    if isinstance(trace_measure, str):
        assert trace_measure in TRACE_MEASURES, ('Must choose a trace measure'
                                                 ' from ' + str(TRACE_MEASURES))
        trace_measure = [trace_measure]
    elif trace_measure is None:
        trace_measure = [trace_measure]
    elif isinstance(trace_measure, list):
        assert all(item in TRACE_MEASURES + [None] for item in trace_measure)
    # Check view_3d, asymptote, save
    assert isinstance(view_3d, bool), 'view_3d must be passed as a bool'
    assert isinstance(asymptote, bool), 'asymptote must be passed as a bool'
    assert isinstance(save, bool), 'save must be passed as a bool'
    multiple = len(systems) > 1
    # Process and plot
    tmp = multiple and asymptote
    asymptote = False if tmp else asymptote
    for idx, sys in enumerate(systems):
        time_evo = sys.time_evolution
        processed = evo.process_evo_data(time_evo, elements, trace_measure)
        times = processed[0]
    return processed



###############################################
###############################################
# parameters combinations
datapath='quantum_HEOM/FMO/fmo_data' 
filename = datapath+'/temp.dat'
col_list = ['temp']
df1 = pd.read_csv(filename, names=col_list, sep="\t+|\s+", engine='python') 

filename = datapath+'/wc.dat'
col_list = ['gamma']
df2 = pd.read_csv(filename, names=col_list, sep="\t+|\s+", engine='python')

filename = datapath+'/lambda.dat'
col_list = ['lambda']
df3 = pd.read_csv(filename, names=col_list, sep="\t+|\s+", engine='python')

sites = 7
timesteps =  200000
states = [1, 6]   # sites with initial excitation
df1['temp'] = df1['temp'].astype(float)
T = df1['temp'].to_numpy()
df2['gamma']=df2['gamma'].astype(float)
wc = df2['gamma'].to_numpy()
df3['lambda']=df3['lambda'].astype(float)
Lambda = df3['lambda'].to_numpy()
#wc = [25,50,75,100,125,150,175,200,225,250,275,300]
#T = [90,110,130,150,170,190,210,230,250,270,290,310]
#Lambda= [10,40,70,100,130,160,190,220,250,280,310]
for initial_state in states:
    for i in range(0, 500):  # number of trajectories
        temperature  = T[i]
        cutoff_freq  = wc[i]
        reorg_energy = Lambda[i]
        print(initial_state, cutoff_freq, reorg_energy, temperature)
        args1 = {'sites': sites,
                'interaction_model': 'FMO',
                'dynamics_model': 'local thermalising lindblad',
                'timesteps': timesteps,
                'cutoff_freq': util.unit_conversion(cutoff_freq, 'fs rad^-1', 'rad ps^-1'),
                'reorg_energy': util.unit_conversion(reorg_energy, 'cm^-1', 'rad ps^-1'),
                'temperature': temperature,
                'deph_rate': 11,
                'init_site_pop': [initial_state],
                }
        # Top plot: initial excitation on site 1, as specified in args1
        q1 = QuantumSystem(**args1)
        processed = plot_dynamics(q1, elements='all')            
        times, matrix_data, squared, distance = processed
        pops = np.array(list(matrix_data.items()), dtype=object)
        data=np.column_stack((times, pops[0,1], pops[1,1], pops[2,1], pops[3,1], pops[4,1], pops[5,1], pops[6,1], pops[7,1], pops[8,1], pops[9,1], pops[10,1], \
                pops[11,1], pops[12,1], pops[13,1], pops[14,1], pops[15,1], pops[16,1], pops[17,1], pops[18,1], pops[19,1], pops[20,1], pops[21,1], pops[22,1], pops[23,1], \
                pops[24,1], pops[25,1], pops[26,1], pops[27,1], pops[28,1], pops[29,1], pops[30,1], pops[31,1], pops[32,1], pops[33,1], pops[34,1], pops[35,1], pops[36,1], \
                pops[37,1], pops[38,1], pops[39,1], pops[40,1], pops[41,1], pops[42,1], pops[43,1], pops[44,1], pops[45,1], pops[46,1], pops[47,1], pops[48,1]))
        filename = str(sites)+"_initial-"+ str(initial_state)+"_wc-" + str(cutoff_freq) + "_lambda-" + str(reorg_energy) + "_temp-" + str(temperature)
        np.save(filename, data)
