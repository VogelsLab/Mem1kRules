# Mem1kRules

This is the companion code to the Memory by a thousand rules https://www.biorxiv.org/content/10.1101/2025.05.28.656584v2.abstract
This code is divided into two parts. 
1/ /results uses pre-generated simulation data to reproduce all figures of the paper. It only requires minimal python libraries and downloading data from a zenodo link.
2/ /cluster_deployment documents how the plastic spiking networks simulations were deployed and parallelized over a few thousand cpu cores in the ISTA cluster, as well as the postprocessing to extract summary statistics from all simulations (regrouped in the dataset available on zenodo). This part requires many more packages to run, and thus follows its own documentation.

The original, full simulation results (>100Tb) are archived at ISTA and can be accessed upon reasonable request.


## How to download the companion data (~2Gb)
1/ run download_data.sh, which will fetch the data for you
OR
2/ click on this link and add this "data" folder at the root of this repo

