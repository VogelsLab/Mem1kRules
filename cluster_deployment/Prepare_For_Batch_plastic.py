import argparse
from os import makedirs, listdir
import numpy as np
import yaml
import h5py
import matplotlib
import os
import pickle
from concurrent import futures
import matplotlib.pyplot as plt

####
import hashlib
def _make_unique_samples(num_samples, prior=None, thetas=None, saved_seeds=[]):
    seeds = []
    if prior is None:
        assert thetas is not None
    else:
        thetas = prior.sample(num_samples)
    new_thetas = []
    for th in thetas:
        str_th = str(th).encode()
        seed = hashlib.md5(str_th).hexdigest()
        if seed not in saved_seeds:
            seeds.append(seed)
            new_thetas.append(th)
    return new_thetas, seeds
####


save_dir = "../data_synapsesbi/seq_IF_EEEIIEII_6pPol/"
runs_path = "../runs_synapsesbi/seq_IF_EEEIIEII_6pPol/"
task_name = "seq_IF_EEEIIEII_6pPol"
# from synapsbi.simulator import Simulator_BND_IF_EEEIIEII_6pPol as Simulator 
from synapsbi.simulator import Simulator_seq_IF_EEEIIEII_6pPol as Simulator
# from synapsbi.simulator import Simulator_BND_CVAIF_EEIE_T4wvceciMLP as Simulator
# from synapsbi.simulator import Simulator_seq_CVAIF_EEIE_T4wvceciMLP as Simulator
# from synapsbi.simulator import Simulator_bg_IF_EEEIIEII_6pPol as Simulator 


####
def make_param_files_cluster(simulator, thetas, seeds, filename):
    """
    simulator: a simulator object as above. TODO make ABC
    """

    with open(filename, "w") as f:
        for th, seed in zip(thetas, seeds):
            rule_str = simulator.rule_str.format(*list(th))
            f.write(simulator.cl_str % (str(seed), rule_str) + '\n')
    print("wrote", len(seeds), "call strings to", filename)

with open("tasks_configs/%s.yaml" % task_name, "r") as f:
	simulator_params = yaml.load(f, Loader=yaml.Loader)
	simulator = Simulator(simulator_params)
####

# thetas = np.load(save_dir + "poly_1639.npz")["thetas"]
# thetas = np.load(save_dir + "poly_2850.npz")["thetas"]
# thetas = np.load(save_dir + "poly_2500.npz")["thetas"]
# thetas = np.load(save_dir + "1k_mf_npe_140225.npy")
thetas = np.load(save_dir + "1k_mf_npe_rho1min2_140325.npy")
# thetas = np.load("../data_synapsesbi/bg_IF_EEEIIEII_6pPol/10k_bg_samples_02082024.npz")["thetas"]
# thetas = np.load(save_dir + "MLP_4565.npz")["thetas"]
# thetas = np.load(save_dir + "MLP_3103.npz")["thetas"]


#### no need to change from there on
num_samples = len(thetas)
thetas, seeds = _make_unique_samples(num_samples, prior=None, thetas=thetas, saved_seeds=[])

parameters = [dict(simulator=simulator, thetas=th.reshape(1, -1), seeds=[seed], return_data=False) for th, seed in zip(thetas, seeds)][:num_samples]

filename = "/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/" + task_name + "_params.txt"
make_param_files_cluster(simulator, thetas, seeds, filename)