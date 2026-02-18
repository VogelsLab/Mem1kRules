from synapsbi.analyse import ComputeMetrics_BND
from synapsbi.utils import save_metric
from typing import List
import numpy as np
import yaml
import h5py
import time
from concurrent import futures

### FOR POLY
metrics = ["rate","rate_i","r_nov","r_fam","std_nov","std_fam","prate", "weef", "weif", "wief", "wiif", "w_blow", "w_creep","cv_isi"]

round_name = "4k_BND_mfnpe_drfamnov0p2_280225"

h5_path = "../data_synapsesbi/BND_IF_EEEIIEII_6pPol/" + str(round_name) + ".h5"

fixed_params = {"break_durations": [1, 9,  10, 40, 60,  180, 300, 600,  2400, 10800],
                "l_pre_test_record": 1}

#### FOR MLP
# metrics = ["rate","rate_i","r_nov","r_fam","std_nov","std_fam","prate", "weef", "wief", "w_blow", "w_creep","cv_isi"]

# round_name = "BND_MLP_3103_1s1h_13022025"
# h5_path = "../data_synapsesbi/BND_CVAIF_EEIE_T4wvceciMLP/" + str(round_name) + ".h5"

# fixed_params = {"break_durations": [1, 9,  10, 40, 60,  180, 300, 600,  2400],
#                 "l_pre_test_record": 1}


def worker_comp_metrics_BND(seeds_list, h5_path, metrics, fixed_params):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        n_keys_to_process = len(seeds_list)
        n_thetas = f[keys[0]]['theta'].shape[0]
        sim_params=dict(f[keys[0]].attrs.items())
        dt = get_dt_metrics(metrics, sim_params, fixed_params, n_thetas)
        data = np.zeros(n_keys_to_process, dtype=dt)
        for i, k in enumerate(seeds_list):
            data[i]['seed'] = k
            data[i]['theta'] = np.array(f[k]["theta"])
            ## we have a blow-up: no spiketimes or weights to gather
            if f[k].attrs["blow_up"] >= 0:
                comp_metrics = ComputeMetrics_BND(spiketimes=None,
                                          sim_params=dict(f[k].attrs.items()),
                                          weights=None,
                                          hard_coded_sim_params=fixed_params)
            else:
                spiketimes = {str(j): f[k]["spiketimes"][str(j)][()] for j in range(0, f[k].attrs["n_recorded"])}
                if f[k].attrs["record_i"]:
                    spiketimes_i = {str(j): f[k]["spiketimes_i"][str(j)][()] for j in range(0, f[k].attrs["n_recorded_i"])}
                else:
                    spiketimes_i = None
                weights = {i: f[k]["weights"][i][()] for i in f[k]["weights"].keys()}
                if weights['t'] is np.NAN:
                    weights = None
                
                comp_metrics = ComputeMetrics_BND(spiketimes=spiketimes,
                                          sim_params=dict(f[k].attrs.items()),
                                          spiketimes_i=spiketimes_i,
                                          weights=weights,
                                          hard_coded_sim_params=fixed_params)
            try:
                for j, metric in enumerate(metrics):
                    data[i][metric] = getattr(comp_metrics, metric)
            except:
                print("exception for seed", k, "assigning -1 for now")
                for j, metric in enumerate(metrics):
                    data[i][metric] = -1
    return(data)

def comp_metrics_BND(h5_path: str, metrics: List[str], parallel=False, n_workers=2, fixed_params=None):
    """
    computes metrics over a list of h5 files. does not check for existence of a previous file
    """
    n_metrics = len(metrics)
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        n_thetas = f[keys[0]]['theta'].shape[0]
        sim_params=dict(f[keys[0]].attrs.items())
        dt = get_dt_metrics(metrics, sim_params, fixed_params, n_thetas)
        print("Found h5 file with", n_keys, "simulations")

    if not parallel:
        data = worker_comp_metrics_BND(keys, h5_path, metrics, fixed_params)

    else:
        data = np.zeros(n_keys, dtype=dt)
        all_seeds_list = np.array_split(keys, n_workers) #divide the keys in equal parts
        with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            jobs = [executor.submit(worker_comp_metrics_BND, seeds_list, h5_path, metrics, fixed_params)\
                    for seeds_list in all_seeds_list]

        data = np.zeros(0, dtype=dt)
        for job in jobs:
            data = np.append(data, job.result(), axis=0)
    return(data)

def get_dt_metrics(metrics, sim_params, fixed_params, n_thetas):
    n_tests = len(fixed_params["break_durations"])
    aux_dt = [('seed', 'U64'), ('theta', np.float64, n_thetas)]
    for metric in metrics:
        if metric in ["rate", "rate_i", "r_nov", "std_nov", "r_fam", "std_fam", "weef", "weif", "wief", "wiif", "w_blow", "w_creep", "cv_isi", "auto_cov"]:
            aux_dt.append( (metric, np.float64, (n_tests) ) )
        elif metric == "prate":
            l_1test = 2*(sim_params["ontime_test"] + sim_params["offtime_test"])
            n_bins_per_test = len(np.arange(-fixed_params["l_pre_test_record"], l_1test, sim_params["bin_size_big"]))-1
            aux_dt.append( (metric, np.float64, (n_tests, n_bins_per_test) ) )
    return(np.dtype(aux_dt))


## Debug mode: check only one seed.    
# with h5py.File(h5_path, "r") as f:
#     keys = list(f.keys())
#     print("inside debug mode of Compute_metrics_BND")
#     sim_params=dict(f[keys[0]].attrs.items())
#     print("sim_params", sim_params)
#     output = worker_comp_metrics_BND([keys[0]], h5_path, metrics, fixed_params)
#     print("seed", output['seed'])
#     print("thetas", output['theta'])
#     print(output)

## Run metrics in parallel
start = time.time()
output = comp_metrics_BND(h5_path, metrics, parallel=True, n_workers=60, fixed_params=fixed_params) #change number of workers
print(time.time()-start, "s")
default_path = h5_path[:-3] + "_metrics.npy"
save_metric(path=default_path,
            data=output)
