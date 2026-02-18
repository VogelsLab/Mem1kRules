# Mem1kRules

This is the companion code to the Memory by a thousand rules https://www.biorxiv.org/content/10.1101/2025.05.28.656584v2.abstract
This code is divided into two parts. 
  
1/ demo_exploring_rules.ipynb is a tutorial which runs without downloading any additional data, and a good entry point to this repo.  
  
2/ "/results" uses pre-generated simulation data (summary statistics from spiking network simulations) to reproduce the analysis and plotting for all figures of the paper. It only requires usual python libraries and downloading the data from a huggingface link ([bconfavr/Mem1kRules](https://huggingface.co/datasets/bconfavr/Mem1kRules)).  
  
3/ "/cluster_deployment" documents how the plastic spiking networks simulations were deployed and parallelized over one thousand cpu cores in the ISTA cluster, as well as the postprocessing to extract the summary statistics (regrouped in the companion dataset). This part requires many more packages to run, and thus follows its own documentation.

Note that the original full simulation results (>100Tb) are archived at ISTA and can be accessed upon reasonable request.

## How to download the companion data (~20Gb)
Download the data on this link and add all files to the "data" folder of this repository: https://huggingface.co/datasets/bconfavr/Mem1kRules

## Libraries needed to run the code
numpy, matplotlib, sklearn, scipy, torch and pandas
You can use the yaml file to create a working conda environment.

## Running spiking networks
This part uses auryn, the code shown in \cluster_deployment builds on the fSBI repository (https://github.com/VogelsLab/fSBI).