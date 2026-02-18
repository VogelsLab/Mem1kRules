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

## TO CHANGE EVERY TIME
array_job_num = 21393255
output_dir = "/nfs/scistore14/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/sim_workdir/"
save_dir = "../data_synapsesbi/seq_IF_EEEIIEII_6pPol/"
runs_path = "../runs_synapsesbi/seq_IF_EEEIIEII_6pPol/"
task_name = "seq_IF_EEEIIEII_6pPol"
round_name = "2500_1s4hBreaks_20082024_bis"
from synapsbi.simulator import Simulator_seq_IF_EEEIIEII_6pPol as Simulator 
#########


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


########################### TO UPDATE ################################################################################
print("Loading samples that have been simulated")
# thetas = np.load(save_dir + "MLP_plausible_bg.npz")["thetas"]
# thetas = np.load(save_dir + "poly_1639.npz")["thetas"]
# thetas = np.load(save_dir + "poly_2850.npz")["thetas"]
thetas = np.load(save_dir + "poly_2500.npz")["thetas"]
# thetas = np.load("../data_synapsesbi/bg_IF_EEEIIEII_6pPol/10k_bg_samples_02082024.npz")["thetas"]
num_samples = len(thetas)
thetas, seeds = _make_unique_samples(num_samples, 
                                     prior=None,
                                     thetas=thetas,
                                     saved_seeds=[])

#num_samples = 100
#theta_0_6pPol = [0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0]
#thetas = np.array([theta_0_6pPol for i in range(num_samples)])
#seeds = ['static'+str(i) for i in range(num_samples)]

print("samples loaded with shape", len(thetas))
################################################################################################################################

print("fetching simulation outputs in", output_dir, "for array job", array_job_num)
outputs = get_output(seeds, array_job_num, output_dir)
print("blown up simulations:", sum(np.array(outputs) > 0), "out of", len(outputs), sum(np.array(outputs) > 0)/len(outputs))

while True:
    try:
        proceed = str(input("Do you wish to create/remove a h5 file for that task? (y/n):"))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

if proceed == "y":
    print("processing each simulation...") #TO CHANGE BELOW                                    ############################################
    save_data_to_hdf5(save_dir + round_name + ".h5",
                      thetas=thetas,
                      seeds=seeds,
                      spiketimes_workdir=simulator_params["workdir"],
                      con_type=simulator_params["con_type"],
                      outputs=outputs,
                      params=simulator_params)
else:
    print("ending processing there")
