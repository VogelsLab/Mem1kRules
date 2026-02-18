import argparse
from os import makedirs, listdir
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

save_dir = "../data_synapsesbi/seq_IF_EEEIIEII_6pPol/"
runs_path = "../runs_synapsesbi/seq_IF_EEEIIEII_6pPol/"
task_name = "seq_IF_EEEIIEII_6pPol" #BND_IF_EEEIIEII_6pPol seq_IF_EEEIIEII_6pPol
# from synapsbi.simulator import Simulator_BND_IF_EEEIIEII_6pPol as Simulator 
from synapsbi.simulator import Simulator_seq_IF_EEEIIEII_6pPol as Simulator

with open("tasks_configs/%s.yaml" % task_name, "r") as f:
	simulator_params = yaml.load(f, Loader=yaml.Loader)
	simulator = Simulator(simulator_params)

num_samples = 100
theta_0_6pPol = [0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0, 0.02, 0.02, 0, 0, 0, 0]
thetas = np.array([theta_0_6pPol for i in range(num_samples)])
seeds = ['static'+str(i) for i in range(num_samples)]

filename = "/nfs/scistore23/vogelgrp/bconfavr/synapsesbi/synapsbi/simulator/cpp_simulators/" + task_name + "_params.txt"
make_param_files_cluster(simulator, thetas, seeds, filename)
