import h5py
import numpy as np


chosen_seed = '868d3f7d4cc437b24f6efd2cc0cdf8da'
# BND_CVAIF_EEIE_T4wvceciMLP BND_IF_EEEIIEII_6pPol seq_IF_EEEIIEII_6pPol

# round_name = "2500_1s4hBreaks_19082024"
# h5_path = "../data_synapsesbi/BND_IF_EEEIIEII_6pPol/" + str(round_name) + ".h5"
# output_file_path = "../data_synapsesbi/BND_IF_EEEIIEII_6pPol/" + round_name + "_" + chosen_seed + ".npy"

# round_name = "4k_seq_1s4hBreaks_mfnpe_dr0p2_28022025" # "2500_seq_1s4hBreaks_16112024" "4k_seq_1s4hBreaks_mfnpe_dr0p2_28022025"
# h5_path = "../data_synapsesbi/seq_IF_EEEIIEII_6pPol/" + str(round_name) + ".h5"
# output_file_path = "../data_synapsesbi/seq_IF_EEEIIEII_6pPol/" + round_name + "_" + chosen_seed + ".npy"

round_name = "seq_MLP_3103_1s1h_27122024" # 
h5_path = "../data_synapsesbi/seq_CVAIF_EEIE_T4wvceciMLP/" + str(round_name) + ".h5"
output_file_path = "../data_synapsesbi/seq_CVAIF_EEIE_T4wvceciMLP/" + round_name + "_" + chosen_seed + ".npy"

with h5py.File(h5_path, "r") as f:
    spiketimes = {str(j): f[chosen_seed]["spiketimes"][str(j)][()] for j in range(0, f[chosen_seed].attrs["n_recorded"])}
    print(list(f[chosen_seed].attrs.items()))
np.save(output_file_path, spiketimes)

print("saved spiketimes for seed", chosen_seed, "to", output_file_path)
