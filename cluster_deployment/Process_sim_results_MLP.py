import argparse
from os import makedirs, listdir
from synapsbi.prior import RestrictedPrior as Prior
from synapsbi.utils import save_data_to_hdf5, read_monitor_weights_files, _make_unique_samples, _forward, get_output_cluster
from synapsbi.simulator import make_param_files_cluster
import numpy as np
import yaml
import h5py
import matplotlib
import os
import pickle
from concurrent import futures
from torch import FloatTensor as FT
import matplotlib.pyplot as plt
import torch

########################### TO UPDATE ################################################################################
array_job_num = 26903587
output_dir = "/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/sim_workdir/"
save_dir = "../data_synapsesbi/BND_CVAIF_EEIE_T4wvceciMLP/"
runs_path = "../runs_synapsesbi/BND_CVAIF_EEIE_T4wvceciMLP/"
task_name = "BND_CVAIF_EEIE_T4wvceciMLP"
round_name = "BND_MLP_3103_1s1h_13022025"
from synapsbi.simulator import Simulator_BND_CVAIF_EEIE_T4wvceciMLP as Simulator 



### TODO
print("Loading samples that have been simulated")
# Choose one appropriate for plastic simulations
# thetas = np.load(save_dir + "MLP_4565.npz")["thetas"]
thetas = np.load(save_dir + "MLP_3103.npz")["thetas"]
num_samples = len(thetas)
thetas, seeds = _make_unique_samples(num_samples, 
                                     prior=None,
                                     thetas=thetas,
                                     saved_seeds=[])
print("samples loaded with shape", len(thetas))
################################################################################################################################

def get_output(seeds, array_job_num, output_dir):
    n_sims = len(seeds)
    output = np.zeros(n_sims)
    count_problems = 0
    for sim_num in range(n_sims):
        seed = seeds[sim_num]
        index_array = sim_num + 1 #array job indexed 1-n_jobs, 1 increment right now
        filename = output_dir + "array_" + str(array_job_num) + "-" + str(index_array) + ".log"
        
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                text = f.read()
                if text.find(str(seed)) < 0:
                    print(filename, "does not seem to be the right file for seed", seed)
                    break
                try:
                    output[sim_num] = float(text.split("cynthia")[1])
                except:
                    print("problem in", filename, "assigning 0 for now")
                    output[sim_num] = 0
                    count_problems += 1
        else:
            print("problem")
    print("total number of problematic files", count_problems)
    return(output)

with open("tasks_configs/%s.yaml" % task_name, "r") as f:
    simulator_params = yaml.load(f, Loader=yaml.Loader)
simulator = Simulator(simulator_params)
print("looking at task", task_name, round_name)
print("looking at directory", simulator_params["workdir"])
print("about to create/update h5 file", save_dir + round_name + ".h5")
print("fetching simulation outputs in", output_dir, "for array job", array_job_num)
outputs = get_output(seeds, array_job_num, output_dir)
print("blown up simulations:", sum(np.array(outputs) > 0), "out of", len(outputs), sum(np.array(outputs) > 0)/len(outputs))

save_data_to_hdf5(save_dir + round_name + ".h5",
                      thetas=thetas,
                      seeds=seeds,
                      spiketimes_workdir=simulator_params["workdir"],
                      con_type=simulator_params["con_type"],
                      outputs=outputs,
                      params=simulator_params)