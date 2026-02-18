import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import re
from scipy.ndimage import gaussian_filter
import pandas as pd
from sklearn.metrics import mean_squared_error


###  Define colors and colormaps of the paper

color_ee = (165/256,42/256,42/256)
color_ei = (242/256, 140/256, 40/256)
color_ie = (8/256, 143/256, 143/256)
color_ii = (47/256, 85/256, 151/256)
color_fam = (142/256,68/256,173/256)
color_nov= (67/256,160/256, 71/256)
color_nov1 = '#50C878'
color_nov2 = '#808000'
color_fam1 = "#884ab2"
color_fam2 = "#ff930a"
color_fam3 = "#f24b04"
color_fam4 = "#d1105a"
color_fam5 = "#471ca8"

gray_shade = 0.90
cdict = {'red':   [[0.0,  color_fam[0], color_fam[0]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_nov[0], color_nov[0]]],
         'green': [[0.0,  color_fam[1],color_fam[1]],
                   [0.5, gray_shade, gray_shade],
                   [1.0,  color_nov[1], color_nov[1]]],
         'blue':  [[0.0,  color_fam[2], color_fam[2]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_nov[2], color_nov[2]]]}
cmap_famdet = LinearSegmentedColormap('cmap_familirity_detection', segmentdata=cdict, N=256)

color_start = (255/256, 255/256, 51/256)
color_end = (166/256, 86/256, 40/256)
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [0.5, gray_shade, gray_shade],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[2], color_end[2]]]}
cmp_transdyn = LinearSegmentedColormap('cmap_transient_dynamics', segmentdata=cdict, N=256)

gray_shade = 0.90
color_start = (55/256, 126/256, 184/256)
color_end = (228/256, 26/256, 28/256)
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [0.5, gray_shade, gray_shade],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[2], color_end[2]]]}

cmp_succrep = LinearSegmentedColormap('cmap_successor_representations', segmentdata=cdict, N=256)

color_start = (166/256, 118/256, 29/256)
color_end = (231/256, 41/256, 138/256)
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [0.5, gray_shade, gray_shade],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_end[2], color_end[2]]]}

cmp_contnov = LinearSegmentedColormap('cmap_contextual_novelty', segmentdata=cdict, N=256)

color_start= color_ee
color_end = (80/256, 80/256, 80/256)
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
newcolors[0, :] = np.array([256/256, 256/256, 256/256, 1])
cmap_ee = ListedColormap(newcolors)

color_start= color_ei
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
newcolors[0, :] = np.array([256/256, 256/256, 256/256, 1])
cmap_ei = ListedColormap(newcolors)

color_start= color_ie
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
newcolors[0, :] = np.array([256/256, 256/256, 256/256, 1])
cmap_ie = ListedColormap(newcolors)

color_start= color_ii
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
newcolors[0, :] = np.array([256/256, 256/256, 256/256, 1])
cmap_ii = ListedColormap(newcolors)

color_start = (256/256, 256/256, 256/256)
color_end = color_ee
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
cmap_ee_white = ListedColormap(newcolors)

color_start = (256/256, 256/256, 256/256)
color_end = color_ei
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
cmap_ei_white = ListedColormap(newcolors)

color_start = (256/256, 256/256, 256/256)
color_end = color_ie
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
cmap_ie_white = ListedColormap(newcolors)

color_start = (256/256, 256/256, 256/256)
color_end = color_ii
cdict = {'red':   [[0.0,  color_start[0], color_start[0]],
                   [1.0,  color_end[0], color_end[0]]],
         'green': [[0.0,  color_start[1],color_start[1]],
                   [1.0,  color_end[1], color_end[1]]],
         'blue':  [[0.0,  color_start[2], color_start[2]],
                   [1.0,  color_end[2], color_end[2]]]}
newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=10000)
aux_cm = newcmp.resampled(10000)
newcolors = aux_cm(np.linspace(0, 1, 10000))
cmap_ii_white = ListedColormap(newcolors)

gray_shade = 0.90
cdict = {'red':   [[0.0,  color_fam[0], color_fam[0]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_nov[0], color_nov[0]]],
         'green': [[0.0,  color_fam[1],color_fam[1]],
                   [0.5, gray_shade, gray_shade],
                   [1.0,  color_nov[1], color_nov[1]]],
         'blue':  [[0.0,  color_fam[2], color_fam[2]],
                   [0.5,  gray_shade, gray_shade],
                   [1.0,  color_nov[2], color_nov[2]]]}
cmp_vis = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)


### General functions for loading data, selecting stable rules

def load_and_merge(save_dir, paths):
    """
    paths: tuple with the files to merge (relative to save_dir). Epxecting a numpy structured array
    """
    n_files = len(paths)
    dataset = np.load(save_dir + paths[0])
    for i in range(1,n_files):
        dataset = np.append(dataset, np.load(save_dir + paths[i]), axis=0)
    print("retrieved", str(len(dataset))+"/"+str(len(np.unique(dataset))) ,"simulations")
    return(dataset)

def get_ind_stable(data):
    # rates [0.1, 50]Hz for all break durations (used to be 0.1)
    cond_rate = np.all(np.logical_and(0.1 <= data["rate"], data["rate"] <= 100), axis=1)
    print(np.sum(cond_rate),"/",len(data), "rules fulfill the rate condition", np.sum(cond_rate)/len(data)*100,"%")

    # cv > 0.7 for all break durations
    cond_cv = np.all((data["cv_isi"] >= 0.7), axis=1)
    print(np.sum(cond_cv),"/",len(data), "rules fulfill the cv condition", np.sum(cond_cv)/len(data)*100,"%")

    # weef weif < 0.5, wief, wiif < 5 after 4h
    cond_wf = np.logical_and( np.logical_and(data["weef"][:,-1] <= 0.5, data["weif"][:,-1] <= 0.5), np.logical_and(data["wief"][:,-1] <= 5, data["wiif"][:,-1] <= 5) )
    print(np.sum(cond_wf),"/",len(data), "rules fulfill the wf condition", np.sum(cond_wf)/len(data)*100,"%")

    # w_blow <= 0.1 after 4h (allowing for weights to do crazy stuff during the task itself), used to be fpr
    cond_wb = data["w_blow"][:,-1] <= 0.1
    print(np.sum(cond_wb),"/",len(data), "rules fulfill the w_blow condition", np.sum(cond_wb)/len(data)*100,"%")

    # # w_creep < 0.2
    # cond_wc = np.logical_and(data["w_creep"][:,-1] <= 0.2, data["w_creep"][:,-2] <= 0.2)
    # print(np.sum(cond_wc),"/",len(data), "rules fulfill the wc condition", np.sum(cond_wc)/len(data)*100,"%")

    cond_all = np.logical_and(np.logical_and(np.logical_and(cond_rate,cond_cv), cond_wf), cond_wb)
    print(np.sum(cond_all),"/",len(data), "rules fulfill all conditions", np.sum(cond_all)/len(data)*100,"%")
    return(cond_all)

def get_ind_stable_pd(data):
    cond_rates = np.all([[r>=0.1 for r in rs] for rs in data['rate']], axis=1)
    print('rates', np.sum(cond_rates),'/',len(cond_rates))

    cond_cvs = np.all([[cv>=0.7 for cv in cvs] for cvs in data['cv_isi']], axis=1)
    print('cv_isi', np.sum(cond_cvs),'/',len(cond_cvs))

    cond_weef = np.all([[wee<=0.5 for wee in wees] for wees in data['weef']], axis=1)
    print('weef', np.sum(cond_weef),'/',len(cond_weef))

    cond_weif = np.all([[wei<=0.5 for wei in weis] for weis in data['weif']], axis=1)
    print('weif', np.sum(cond_weif),'/',len(cond_weif))

    cond_wief = np.all([[wie<=5 for wie in wies] for wies in data['wief']], axis=1)
    print('wief', np.sum(cond_wief),'/',len(cond_wief))

    cond_wiif = np.all([[wii<=5 for wii in wiis] for wiis in data['wiif']], axis=1)
    print('wiif', np.sum(cond_wiif),'/',len(cond_wiif))

    cond_wb = np.array([wblows[-1]<=0.1 for wblows in data['w_blow']])
    print('w_blow', np.sum(cond_wb),'/',len(cond_wb))

    inds_stability_tot = np.logical_and(np.logical_and(np.logical_and(cond_rates, cond_cvs), np.logical_and(cond_weef, cond_weif)),np.logical_and(np.logical_and(cond_wief, cond_wiif),cond_wb))
    print(np.sum(inds_stability_tot),"/",len(data), "rules fulfill all conditions", np.sum(inds_stability_tot)/len(data)*100,"%")
    return(inds_stability_tot)

def get_ind_stable_MLP(data):
    # rates [0.5, 50]Hz for all break durations
    cond_rate = np.all(np.logical_and(0.5 <= data["rate"], data["rate"] <= 100), axis=1)
    print(np.sum(cond_rate),"/",len(data), "rules fulfill the rate condition", np.sum(cond_rate)/len(data)*100,"%")

    # cv > 0.7 for all break durations
    cond_cv = np.all((data["cv_isi"] >= 0.7), axis=1)
    print(np.sum(cond_cv),"/",len(data), "rules fulfill the cv condition", np.sum(cond_cv)/len(data)*100,"%")

    # weef weif < 0.5, wief, wiif < 5 after 4h
    cond_wf = np.logical_and( data["weef"][:,-1] <= 0.5, data["wief"][:,-1] <= 5)
    print(np.sum(cond_wf),"/",len(data), "rules fulfill the wf condition", np.sum(cond_wf)/len(data)*100,"%")

    # w_blow <= 0.1 for all break durations
    cond_wb = np.all((data["w_blow"] <= 0.1), axis=1)
    print(np.sum(cond_wb),"/",len(data), "rules fulfill the w_blow condition", np.sum(cond_wb)/len(data)*100,"%")

    # w_creep < 0.2
    cond_wc = np.logical_and(data["w_creep"][:,-1] <= 0.2, data["w_creep"][:,-2] <= 0.2)
    print(np.sum(cond_wc),"/",len(data), "rules fulfill the wc condition", np.sum(cond_wc)/len(data)*100,"%")

    cond_all = np.logical_and(np.logical_and(np.logical_and(np.logical_and(cond_rate,cond_cv), cond_wf), cond_wb), cond_wc)
    print(np.sum(cond_all),"/",len(data), "rules fulfill all conditions", np.sum(cond_all)/len(data)*100,"%")
    return(cond_all)

def get_shape_histogram(rules, x_lim=[-0.2,0.2], n_timebins=1000, n_dwbins=500):
    n_rules = len(rules)
    
    # 1/ get the rule's response to the pre post protocol
    ts = torch.tile(torch.linspace(x_lim[0], x_lim[1], n_timebins), (n_rules,1)) # size [n_rules, n_timebins]
    ind_t_pos = 0 #find where dt changes sign, that's when we switch which synaptic trace is non zero
    while ts[0, ind_t_pos] < 0:
        ind_t_pos += 1

    if type(rules) == np.ndarray:
        rules = torch.Tensor(rules)
    rules_tiled = torch.tile(rules, (n_timebins, 1, 1)).T # size [n_rules, n_timebins]

    dws_ee = np.zeros( (n_rules, n_timebins) )
    dws_ei = np.zeros( (n_rules, n_timebins) )
    dws_ie = np.zeros( (n_rules, n_timebins) )
    dws_ii = np.zeros( (n_rules, n_timebins) )

    dws_ee[:,:ind_t_pos] = rules_tiled[2,:,:ind_t_pos] + rules_tiled[3,:,:ind_t_pos] + torch.mul(rules_tiled[5,:,:ind_t_pos], torch.exp(torch.div(-torch.abs(ts[:,:ind_t_pos]), rules_tiled[1,:,:ind_t_pos])))
    dws_ee[:,ind_t_pos:] = rules_tiled[2,:,ind_t_pos:] + rules_tiled[3,:,ind_t_pos:] + torch.mul(rules_tiled[4,:,ind_t_pos:], torch.exp(torch.div(-torch.abs(ts[:,ind_t_pos:]), rules_tiled[0,:,ind_t_pos:])))
    
    dws_ei[:,:ind_t_pos] = rules_tiled[2+6,:,:ind_t_pos] + rules_tiled[3+6,:,:ind_t_pos] + torch.mul(rules_tiled[5+6,:,:ind_t_pos], torch.exp(torch.div(-torch.abs(ts[:,:ind_t_pos]), rules_tiled[1+6,:,:ind_t_pos])))
    dws_ei[:,ind_t_pos:] = rules_tiled[2+6,:,ind_t_pos:] + rules_tiled[3+6,:,ind_t_pos:] + torch.mul(rules_tiled[4+6,:,ind_t_pos:], torch.exp(torch.div(-torch.abs(ts[:,ind_t_pos:]), rules_tiled[0+6,:,ind_t_pos:])))

    dws_ie[:,:ind_t_pos] = rules_tiled[2+2*6,:,:ind_t_pos] + rules_tiled[3+2*6,:,:ind_t_pos] + torch.mul(rules_tiled[5+2*6,:,:ind_t_pos], torch.exp(torch.div(-torch.abs(ts[:,:ind_t_pos]), rules_tiled[1+2*6,:,:ind_t_pos])))
    dws_ie[:,ind_t_pos:] = rules_tiled[2+2*6,:,ind_t_pos:] + rules_tiled[3+2*6,:,ind_t_pos:] + torch.mul(rules_tiled[4+2*6,:,ind_t_pos:], torch.exp(torch.div(-torch.abs(ts[:,ind_t_pos:]), rules_tiled[0+2*6,:,ind_t_pos:])))

    dws_ii[:,:ind_t_pos] = rules_tiled[2+3*6,:,:ind_t_pos] + rules_tiled[3+3*6,:,:ind_t_pos] + torch.mul(rules_tiled[5+3*6,:,:ind_t_pos], torch.exp(torch.div(-torch.abs(ts[:,:ind_t_pos]), rules_tiled[1+3*6,:,:ind_t_pos])))
    dws_ii[:,ind_t_pos:] = rules_tiled[2+3*6,:,ind_t_pos:] + rules_tiled[3+3*6,:,ind_t_pos:] + torch.mul(rules_tiled[4+3*6,:,ind_t_pos:], torch.exp(torch.div(-torch.abs(ts[:,ind_t_pos:]), rules_tiled[0+3*6,:,ind_t_pos:])))

    # 2/ make a histogram of dws across rules, one per time bin
    # normalize rule responses first (get rid of learning rate effects)
    dws_norm_ee = np.zeros( (n_rules, n_timebins) )
    dws_norm_ei = np.zeros( (n_rules, n_timebins) )
    dws_norm_ie = np.zeros( (n_rules, n_timebins) )
    dws_norm_ii = np.zeros( (n_rules, n_timebins) )
    for i in range(n_rules):
        dws_norm_ee[i,:] = dws_ee[i,:]/np.max(np.abs(dws_ee[i,:]))
        dws_norm_ei[i,:] = dws_ei[i,:]/np.max(np.abs(dws_ei[i,:]))
        dws_norm_ie[i,:] = dws_ie[i,:]/np.max(np.abs(dws_ie[i,:]))
        dws_norm_ii[i,:] = dws_ii[i,:]/np.max(np.abs(dws_ii[i,:]))

    # 3/ make a histogram of dws across rules, one per time bin
    dws_hist_norm_ee = np.zeros( (n_timebins, n_dwbins) )
    dws_hist_norm_ei = np.zeros( (n_timebins, n_dwbins) )
    dws_hist_norm_ie = np.zeros( (n_timebins, n_dwbins) )
    dws_hist_norm_ii = np.zeros( (n_timebins, n_dwbins) )
    for i in range(n_timebins):
        dws_hist_aux_norm_ee,_ = np.histogram(dws_norm_ee[:,i], bins=n_dwbins, range=[-1, 1])
        dws_hist_norm_ee[i,:] = dws_hist_aux_norm_ee

        dws_hist_aux_norm_ei,_ = np.histogram(dws_norm_ei[:,i], bins=n_dwbins, range=[-1, 1])
        dws_hist_norm_ei[i,:] = dws_hist_aux_norm_ei

        dws_hist_aux_norm_ie,_ = np.histogram(dws_norm_ie[:,i], bins=n_dwbins, range=[-1, 1])
        dws_hist_norm_ie[i,:] = dws_hist_aux_norm_ie

        dws_hist_aux_norm_ii,_ = np.histogram(dws_norm_ii[:,i], bins=n_dwbins, range=[-1, 1])
        dws_hist_norm_ii[i,:] = dws_hist_aux_norm_ii
    _,dw_bins_norm = np.histogram(dws_norm_ee[:,0], bins=n_dwbins, range=[-1, 1])

    return(dws_ee, dws_ei, dws_ie, dws_ii, dws_hist_norm_ee, dws_hist_norm_ei, dws_hist_norm_ie, dws_hist_norm_ii)

def N_AND(conditions):
    #condition list of conditions
    n_conditions = len(conditions)
    cond = conditions[0]
    for i in range(1, n_conditions):
        cond = np.logical_and(cond, conditions[i])
    return(cond)

def N_OR(conditions):
    #condition list of conditions
    n_conditions = len(conditions)
    cond = conditions[0]
    for i in range(1, n_conditions):
        cond = np.logical_or(cond, conditions[i])
    return(cond)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Truncate a colormap to a subrange [minval, maxval]."""
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


### General plotting

def plot_metric_tbreaks_all_rules(metric, x_last_significant, figsize=(3,2), dpi=600, cmap = "Spectral", vmin=-0.5, vmax=0.5,
                                   x_lim = None, x_ticks = None, x_ticklabels = None, heatmap_label="", cbarhandlepad=None, cbarticklabels=None,
                                   x_label=None, y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None,
                                   axwidth=1.5, linewidth=0.01, xticks_pad=5, yticks_pad=0, labelpad_xlabel=5, cbarticks=None,
                                   rotation=0, labelpad_ylabel=0, color_ylabel='black', fontsize=10, font='Arial', linewidth_sign=0.2):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    matrice = ax.imshow(metric, vmin=vmin, vmax=vmax, cmap=cmap, aspect=len(metric[0])/(len(metric)*figsize[0]/figsize[1]))
    ax.plot(x_last_significant+0.5, [i for i in range(len(x_last_significant))], marker='none', color='black', linewidth=linewidth_sign)
    # ax.scatter(x_last_significant+0.5, [i for i in range(len(x_last_significant))], marker='.', color='black', s=0.2, edgecolor='none')

    if x_ticks is not None:
        ax.set_xticks(np.array(x_ticks)+0.5)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)

    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel, color=color_ylabel)
    
    cbar = fig.colorbar(matrice, label=heatmap_label,  ax=ax, pad=0.06, shrink=0.84) #aspect=18,
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(axwidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels != None:
        cbar.set_ticklabels(cbarticklabels)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)

    ax.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    plt.show()

def plot_pop_rate(rs=None, ts=None, t_lim=None, r_lim=None, color="#008080", x_label=None, y_label=r'$r_{pop} \; (Hz)$', 
save_path=None, ax=None, linewidth=3, axwidth=3, fontsize=20, figsize=(5, 2), font = "arial", rotation=0, dpi=200, 
x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, target=None, x_milestones=None, linewidth_milestones=1.5, alpha_milestone=1):
    """
    makes a plot of the popualtion firing rate across time

    args: rs np.array(float)
          ts np.array(float) same size as rs
          t_lim = [t_min,t_max] float: in seconds, otherwise "default"
          r_lim = [r_min, r_max] float: Hz, otherwise "default"
          y_label str: will be displayed on the y axis of the plot
          save_path str: if not "" will same the figure
          other args are cosmetic kwargs from matplotlib, see matplotlib documentation

    returns: matplotlib axis
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if x_milestones is not None:
        for x in x_milestones:
            ax.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone)

    ax.plot(ts, rs, color=color, linewidth=linewidth, marker='')

    if target is not None:
        ax.hlines(target,t_lim[0],t_lim[1], linestyles="--", linewidth=linewidth, color=color, zorder=0)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation=rotation)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    if t_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
    
    if r_lim is not None:
        ax.set_ylim([r_lim[0], r_lim[1]])
        ax.set_yticks([r_lim[0], r_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    return(ax)

def plot_raster_w_engrams(sts, neuron_labels=None, n_recorded=None, t_start_each_stim = None,
                          x_lim=None, y_lim=None, fontsize=20, label_colors=None, ontime=None,
                x_label="", y_label="", markersize=0.05, figsize=(20, 3), linewidth_stim_line=3,
                font="arial", x_ticks=None, x_ticklabels=None, y_ticks=None, y_stim_line=None,
                y_ticklabels=None, tickwidth=2, axwidth=3, dpi=200, ylabel_xloc=-0.1,
                ylabel_yloc=0.15, xlabel_xloc=0.4, xlabel_yloc=-0.1, y_bar_xloc=-0.1, y_bar_ylocs=[2.85/7, 3.6/7], cartoon=False):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ct = 0
    for label_num in range(0,8,+1):
        neurons_to_plot = np.argwhere(neuron_labels==label_num).flatten()
        # print(label_num, neurons_to_plot.shape, neurons_to_plot)
        for neuron_num in neurons_to_plot:
            ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=label_colors[label_num], s=markersize, edgecolors=None, marker='o')
            ct += 1

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if (x_lim is None) and (x_ticks is not None):
        ax.set_xlim([x_ticks[0], x_ticks[-1]])
    if x_ticks is None:
        ax.set_xticks([x_lim[0], x_lim[1]])
    else:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)

    for i,t in enumerate(t_start_each_stim):
        ax.plot([t,t+ontime], [y_stim_line,y_stim_line], color=label_colors[i], linewidth=linewidth_stim_line,
                solid_capstyle="butt")

    if x_label is not None:
        fig.text(xlabel_xloc, xlabel_yloc, x_label, fontsize=fontsize, fontname=font, ha='center')

    if y_label is not None:
        ax.plot([y_bar_xloc, y_bar_xloc], y_bar_ylocs, transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(ylabel_xloc, ylabel_yloc, y_label, fontsize=fontsize, fontname=font, rotation=90, ha='center')

    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim([-1, n_recorded + 0.1])
    if y_ticks is None:
        ax.set_yticks([0, n_recorded])
    else:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=tickwidth, labelsize=fontsize, length=tickwidth*2, pad = 10)
    ax.tick_params(axis='y', pad = 0)

    if cartoon:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig("cartoon_raster.png")

    return

def plot_4_rules(thetas,
                n_bins=1000,
                x_lim=[-0.2,0.2],
                y_lim = [-1,1],
                y_ticks=[],
                x_ticks=[],
                x_ticklabels=None,
                x_label=r'$\Delta t$',
                y_label=r'$\Delta w$',
                color_ee=(165/256,42/256,42/256),
                color_ei=(242/256, 140/256, 40/256),
                color_ie=(8/256, 143/256, 143/256),
                color_ii=(47/256, 85/256, 151/256),
                color_ylabel="black",
                figsize=(0.6,0.1),
                labelpad_xlabel=1,
                fontsize=10,
                labelpad_ylabel=27,
                linewidth=0.8,
                axwidth=0.8,
                dpi=600,
                xticks_pad=0,
                yticks_pad=0,
                rotation=0,
                font='arial',
                y_ticklabels=None,
                save_fig=False,
                path=""):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    ts = np.linspace(x_lim[0], x_lim[1],num=n_bins)
    ind_t_pos = 0
    while ts[ind_t_pos] < 0:
        ind_t_pos += 1

    dws = np.array([thetas[2] + thetas[3] + thetas[5]*np.exp(-np.abs(ts[i])/thetas[1]) for i in range(ind_t_pos)])
    dws = np.append(dws, np.array([thetas[2] + thetas[3] + thetas[4]*np.exp(-np.abs(ts[i])/thetas[0]) for i in range(ind_t_pos, len(ts))]), axis=0)
    line1_1, = ax1.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ee, linewidth=linewidth)
    line1_2, = ax1.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ee, linewidth=linewidth)
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.set_ylim([-1.15*np.max(np.abs(dws)), 1.15*np.max(np.abs(dws))])
    ax1.set_yticks(y_ticks)
    # ax1.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel, color=color_ylabel)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(axwidth)
    ax1.spines['left'].set_linewidth(axwidth)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['left'].set_position('zero')
    ax1.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax1.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line1_1.set_solid_capstyle('round')
    line1_2.set_solid_capstyle('round')

    dws = np.array([thetas[2+6] + thetas[3+6] + thetas[5+6]*np.exp(-np.abs(ts[i])/thetas[1+6]) for i in range(ind_t_pos)])
    dws = np.append(dws, np.array([thetas[2+6] + thetas[3+6] + thetas[4+6]*np.exp(-np.abs(ts[i])/thetas[0+6]) for i in range(ind_t_pos, len(ts))]), axis=0)
    # line2 = ax2.plot(ts, dws, color=color_ei, linewidth=linewidth)
    line2_1, = ax2.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ei, linewidth=linewidth)
    line2_2, = ax2.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ei, linewidth=linewidth)
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.set_ylim([-1.15*np.max(np.abs(dws)), 1.15*np.max(np.abs(dws))])
    ax2.set_yticks(y_ticks)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(axwidth)
    ax2.spines['left'].set_linewidth(axwidth)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['left'].set_position('zero')
    ax2.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax2.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line2_1.set_solid_capstyle('round')
    line2_2.set_solid_capstyle('round')

    dws = np.array([thetas[2+2*6] + thetas[3+2*6] + thetas[5+2*6]*np.exp(-np.abs(ts[i])/thetas[1+2*6]) for i in range(ind_t_pos)])
    dws = np.append(dws, np.array([thetas[2+2*6] + thetas[3+2*6] + thetas[4+2*6]*np.exp(-np.abs(ts[i])/thetas[0+2*6]) for i in range(ind_t_pos, len(ts))]), axis=0)
    # line3 = ax3.plot(ts, dws, color=color_ie, linewidth=linewidth)
    line3_1, = ax3.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ie, linewidth=linewidth)
    line3_2, = ax3.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ie, linewidth=linewidth)
    ax3.set_xlim(x_lim)
    ax3.set_xticks(x_ticks)
    ax3.set_ylim([-1.15*np.max(np.abs(dws)), 1.15*np.max(np.abs(dws))])
    ax3.set_yticks(y_ticks)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(axwidth)
    ax3.spines['left'].set_linewidth(axwidth)
    ax3.spines['bottom'].set_position('zero')
    ax3.spines['left'].set_position('zero')
    ax3.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax3.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line3_1.set_solid_capstyle('round')
    line3_2.set_solid_capstyle('round')

    dws = np.array([thetas[2+3*6] + thetas[3+3*6] + thetas[5+3*6]*np.exp(-np.abs(ts[i])/thetas[1+3*6]) for i in range(ind_t_pos)])
    dws = np.append(dws, np.array([thetas[2+3*6] + thetas[3+3*6] + thetas[4+3*6]*np.exp(-np.abs(ts[i])/thetas[0+3*6]) for i in range(ind_t_pos, len(ts))]), axis=0)
    # line4 = ax4.plot(ts, dws, color=color_ii, linewidth=linewidth)
    line4_1, = ax4.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ii, linewidth=linewidth)
    line4_2, = ax4.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ii, linewidth=linewidth)
    ax4.set_xlim(x_lim)
    ax4.set_xticks(x_ticks)
    ax4.set_ylim([-1.15*np.max(np.abs(dws)), 1.15*np.max(np.abs(dws))])
    ax4.set_yticks(y_ticks)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_linewidth(axwidth)
    ax4.spines['left'].set_linewidth(axwidth)
    ax4.spines['bottom'].set_position('zero')
    ax4.spines['left'].set_position('zero')
    ax4.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax4.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line4_1.set_solid_capstyle('round') #'butt', 'projecting', 'round'
    line4_2.set_solid_capstyle('round')

    # fig.text(0.1, 0.05, x_label, ha='center')
    # fig.text(0.05, 0.6, y_label, va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.2, hspace=0)
    if save_fig:
        plt.savefig(path, transparent=True)
    plt.show()

def plot_4_rules_histogram(dws_hist_ee,
                           dws_hist_ei,
                           dws_hist_ie,
                           dws_hist_ii,
                           cmap_list = ['viridis', 'viridis', 'viridis', 'viridis'],
                            # x_lim=[-0.2,0.2],
                            # y_lim = [-1,1],
                            cap = 25,
                            y_ticks=[],
                            x_ticks=[],
                            x_ticklabels=None,
                            x_label=r'$\Delta t$',
                            y_label=r'$\Delta w$',
                            color_ee=(165/256,42/256,42/256),
                            color_ei=(242/256, 140/256, 40/256),
                            color_ie=(8/256, 143/256, 143/256),
                            color_ii=(47/256, 85/256, 151/256),
                            color_ylabel="black",
                            figsize=(0.6,0.1),
                            labelpad_xlabel=1,
                            fontsize=10,
                            labelpad_ylabel=27,
                            linewidth=0.8,
                            axwidth=0.8,
                            dpi=600,
                            xticks_pad=0,
                            yticks_pad=0,
                            rotation=0,
                            font='arial',
                            y_ticklabels=None,
                            save_fig=False,
                            path=""):
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    n_dwbins = dws_hist_ee.shape[1]
    n_timebins = dws_hist_ee.shape[0]
    
    ax1.imshow(np.flip(dws_hist_ee, axis=1).T, vmin=0, vmax=cap, aspect=(n_timebins)/(n_dwbins)*figsize[0]/figsize[1]/4, cmap=cmap_list[0])
    ax1.axvline(x=n_timebins//2, ymin=-1, ymax=1, linewidth=linewidth, color='black')
    ax1.plot([0,n_timebins-1], [n_dwbins//2,n_dwbins//2], linewidth=linewidth, color='black')
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax1.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    ax2.imshow(np.flip(dws_hist_ei, axis=1).T, vmin=0, vmax=cap, aspect=(n_timebins)/(n_dwbins)*figsize[0]/figsize[1]/4, cmap=cmap_list[1])
    ax2.axvline(x=n_timebins//2, ymin=-1, ymax=1, linewidth=linewidth, color='black')
    ax2.plot([0,n_timebins-1], [n_dwbins//2,n_dwbins//2], linewidth=linewidth, color='black')
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax2.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    ax3.imshow(np.flip(dws_hist_ie, axis=1).T, vmin=0, vmax=cap, aspect=(n_timebins)/(n_dwbins)*figsize[0]/figsize[1]/4, cmap=cmap_list[2])
    ax3.axvline(x=n_timebins//2, ymin=-1, ymax=1, linewidth=linewidth, color='black')
    ax3.plot([0,n_timebins-1], [n_dwbins//2,n_dwbins//2], linewidth=linewidth, color='black')
    ax3.set_xticks(x_ticks)
    ax3.set_yticks(y_ticks)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax3.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    ax4.imshow(np.flip(dws_hist_ii, axis=1).T, vmin=0, vmax=cap, aspect=(n_timebins)/(n_dwbins)*figsize[0]/figsize[1]/4, cmap=cmap_list[3])
    ax4.axvline(x=n_timebins//2, ymin=-1, ymax=1, linewidth=linewidth, color='black')
    ax4.plot([0,n_timebins-1], [n_dwbins//2,n_dwbins//2], linewidth=linewidth, color='black')
    ax4.set_xticks(x_ticks)
    ax4.set_yticks(y_ticks)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax4.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    plt.subplots_adjust(wspace=0.2, hspace=0)
    if save_fig:
        plt.savefig(path, transparent=True)
    plt.show()

def plot_fraction_significant_rules(prop, figsize=(1,1), dpi=600, x_lim=None, x_ticks=None, x_ticklabels=None,
                                   x_label=None, y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None,
                                   axwidth=1.5, linewidth=1.5, xticks_pad=5, yticks_pad=0, labelpad_xlabel=5,
                                   labelpad_ylabel=0, color='black', fontsize=10, font='Arial'):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(prop, linewidth=linewidth, color=color, clip_on=False)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)

    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)

    ax.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    plt.show()

def plot_input_raster(sts, neuron_indices, t_lim=None, fontsize=10, color="black", x_label="", y_label="", 
                      markersize=0.05, figsize=(20, 3), font="arial", x_ticks=None, x_ticklabels=None, x_tickspad=0,
                      y_ticks=None, y_ticklabels=None, y_tickspad=0, tickwidth=2, dpi=600):
    """
    Make a raster plot from an array of spike times

    args: sts dict key=neuron_id value = np.array(float) spiketimes
          neurons_ids list(int)
          t_lim = [t_min,t_max] float in seconds OR "default" 
          save_path str: if not "" will same the figure
          other args are cosmetic kwargs from matplotlib, see matplotlib documentation

    returns: matplotlib axis
    """
    n_to_plot = len(neuron_indices)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ct = 0
    for neuron_num in neuron_indices:
        # ax.plot(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linestyle='', marker='.',color=color, markersize=markersize)
        ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=color, s=markersize, edgecolors=None, marker='.')
        ct += 1
    if t_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
    if (t_lim is None) and (x_ticks is not None):
        ax.set_xlim([x_ticks[0], x_ticks[-1]])
    if x_ticks is None:
        ax.set_xticks([t_lim[0], t_lim[1]])
    else:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    
    ax.set_ylim([0, n_to_plot + 0.1])
    if y_ticks is None:
        ax.set_yticks([0, n_to_plot])
    else:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=tickwidth, labelsize=fontsize, length=tickwidth*2, pad=x_tickspad)
    ax.tick_params(axis='y', pad=y_tickspad)
    plt.show()
    return()

def plot_engram_matrix(w=None, pre_engrams=None, post_engrams=None, 
                       engrams_to_plot=None,
                       N_familiar=None, dpi=600,
                       labels=['n1', 'n2', 'f1', 'f2', 'f3', 'f4', 'f5'],
                       mean_weight=None,
                v_lim = None, figsize=(5,5), save_path=None, cmap="cividis",xlabel="post", xlabelpad=0, xticklabelpad=0,
                ylabel="pre", ylabelpad=0, yticklabelpad=0,
                axwidth=2, font = "Arial", fontsize=15, ax=None, cbarticks=None, cbarticklabels=[''], cbarlabel=None): 
    
    # Making matrix
    n_patterns = len(engrams_to_plot)
    # x = ["n"+str(i+1) for i in range(n_patterns-N_familiar)] + ["f"+str(i+1) for i in range(N_familiar)]
    
    z=np.zeros((n_patterns, n_patterns))
    for i, pre_stim in enumerate(engrams_to_plot):
        for j, post_stim in enumerate(engrams_to_plot):
            z[i,j] = get_mean_weight(w, pre_engrams[str(pre_stim)], post_engrams[str(post_stim)])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if v_lim is None:
        v_lim = [np.min(z), np.max(z)]
    print(np.min(z), np.mean(z), np.max(z))

    if mean_weight is not None:
        dmax = np.max(np.abs(z-mean_weight))
        v_lim = [np.max([mean_weight-dmax,0]), mean_weight+dmax]
        cbarticks = [np.max([mean_weight-dmax,0]), mean_weight, mean_weight+dmax]
        cbarticklabels = [str(np.round(np.max([mean_weight-dmax,0]), 2)), cbarticklabels[0], str(np.round(mean_weight+dmax,2))]
        if mean_weight-dmax < 0: #weights can't be egative, truncate teh colormap so that mean weight is still white
            cmap = truncate_colormap(cmap, minval= 0.5 - mean_weight/dmax/2, maxval=1)
            # print(cbarticks, mean_weight/dmax, dmax)

    ax.set_xlabel(xlabel, fontsize=fontsize, fontname=font, labelpad=xlabelpad)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(ylabel, fontsize=fontsize, fontname=font, labelpad=ylabelpad)
    
    matrice = ax.imshow(z, vmin=v_lim[0], vmax=v_lim[1], cmap=plt.get_cmap(cmap))
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', which='major', pad=xticklabelpad)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', which='major', pad=yticklabelpad)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(matrice, cax=cax, drawedges=False)
    if cbarticks is not None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.ax.set_yticklabels(cbarticklabels)
        cbar.ax.yaxis.set_tick_params(pad=1)
    cbar.set_label(label=cbarlabel, size=fontsize)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(axwidth)
    cbar.ax.tick_params(labelsize=fontsize, width=axwidth, length=2*axwidth)
    
    ax.spines['top'].set_linewidth(axwidth)
    ax.spines['right'].set_linewidth(axwidth)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth, top=True, bottom=False, labeltop=True, labelbottom=False)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if save_path!=None and ax != None: 
        fig.savefig(save_path+ ".png", format='png', dpi=800, transparent=True, bbox_inches='tight')
    plt.show()
    return()

def plot_raster_w_engrams_sep_background(sts, neuron_labels=None, n_recorded=None, t_start_each_stim = None, n_tot_stim=7,
                          x_lim=None, y_lim=None, fontsize=20, colors_raster=None, colors_label=None, ontime=None,
                x_label="", y_label="", markersize=0.05, figsize=(20, 3), linewidth_stim_line=3,
                font="arial", x_ticks=None, x_ticklabels=None, y_ticks=None, y_stim_line=None, lag_engr_bg=20, lag_engr_engr=5,
                y_ticklabels=None, tickwidth=2, axwidth=3, dpi=200, ylabel_xloc=-0.1,
                ylabel_yloc=0.15, xlabel_xloc=0.4, xlabel_yloc=-0.1, y_bar_xloc=-0.1, y_bar_ylocs=[2.85/7, 3.6/7], cartoon=False):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ct = 0
    for label_num in range(0,5,+1):
        neurons_to_plot = np.argwhere(neuron_labels==label_num).flatten()
        for neuron_num in neurons_to_plot:
            ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=colors_raster[label_num], s=markersize, edgecolors=None, marker='o')
            ct += 1
        ct += lag_engr_engr
    
    # plot background activity
    neurons_to_plot = np.argwhere(neuron_labels==5).flatten()
    ct += lag_engr_bg
    for neuron_num in neurons_to_plot:
        ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=colors_raster[5], s=markersize, edgecolors=None, marker='o')
        ct += 1


    if x_lim is not None:
        ax.set_xlim(x_lim)
    if (x_lim is None) and (x_ticks is not None):
        ax.set_xlim([x_ticks[0], x_ticks[-1]])
    if x_ticks is None:
        ax.set_xticks([x_lim[0], x_lim[1]])
    else:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)

    for i,t in enumerate(t_start_each_stim):
        ax.plot([t,t+ontime], [y_stim_line+lag_engr_bg+4*lag_engr_engr,y_stim_line+lag_engr_bg+4*lag_engr_engr], color=colors_label[i%n_tot_stim], linewidth=linewidth_stim_line,
                solid_capstyle="butt")

    if x_label is not None:
        fig.text(xlabel_xloc, xlabel_yloc, x_label, fontsize=fontsize, fontname=font, ha='center')

    if y_label is not None:
        ax.plot([y_bar_xloc, y_bar_xloc], y_bar_ylocs, transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(ylabel_xloc, ylabel_yloc, y_label, fontsize=fontsize, fontname=font, rotation=90, ha='center')

    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim([-1, n_recorded + 0.1])
    if y_ticks is None:
        ax.set_yticks([0, n_recorded])
    else:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=tickwidth, labelsize=fontsize, length=tickwidth*2, pad = 10)
    ax.tick_params(axis='y', pad = 0)

    if cartoon:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig("cartoon_raster.png")

    return

def plot_significance_2t_1metric(data1_dict, data2_dict,
                    xlabel, ylabel,
                        font = "Arial", 
                        fontsize = 10, 
                        linewidth = 1.5, 
                        xlim = None, 
                        ylim = None,
                        figsize=(1.5, 1),
                        xticks=None,
                        yticks=None,
                        xhandlepad=None,
                        yhandlepad=None,
                        s=1, #can be a list too for group specific sizes
                        edgecolors=None, #can be a list too for group specific sizes
                        linewidth_marker=0.0, #can be a list too for group specific sizes
                        center_axes = False,
                        dpi=600,
                        colors='black', #can be a list too for group specific sizes, 'none' for hollow
                        color_xlabel="black",
                        color_ylabel="black",
                        marker='o', #can be a list too for group specific sizes
                        linewidth_line=1.5,
                        color_line="black",
                        alpha_line=0.5,
                        xticklabels=None,
                        yticklabels=None,
                        zorder_line=0.1,
                        which_line = "all",
                        labels=["plastic","static"],
                        bbox_to_anchor=(1,1),
                        labelspacing=1,
                        handletextpad=0):

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.subplot()

    n_data = len(data1_dict.keys())
    print(n_data)

    if not isinstance(s, list):
        s = [s for i in range(n_data)]
    if not isinstance(edgecolors, list):
        edgecolors = [edgecolors for i in range(n_data)]
    if not isinstance(linewidth_marker, list):
        linewidth_marker = [linewidth_marker for i in range(n_data)]
    if not isinstance(marker, list):
        marker = [marker for i in range(n_data)]
    if not isinstance(colors, list):
        colors = [colors for i in range(n_data)]
    
    
    img = ax.scatter(data1_dict['0'], data2_dict['0'], s=s[0], 
                     marker=marker[0], edgecolors=edgecolors[0], 
                     facecolors=colors[0], linewidths=linewidth_marker[0], zorder=10, label=labels[0])

    for ind in range(1, n_data):
        img = ax.scatter(data1_dict[str(ind)], data2_dict[str(ind)], s=s[ind], 
                         marker=marker[ind], edgecolors=edgecolors[ind], 
                     facecolors=colors[ind], linewidths=linewidth_marker[ind], zorder=10, label=labels[ind])
        
    if (xlim is not None) and (ylim is not None):
        start=min(xlim[0], ylim[0])
        stop=max(xlim[1], ylim[1])

    if which_line == "all" or which_line == "diag":
        ax.plot([start, stop], [start, stop], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)

    if which_line == "all" or which_line == "vh":
        ax.plot([1, 1], [start, stop], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
        ax.plot([start, stop], [1, 1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)

    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth, pad=2)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticks != None:
        ax.set_yticks(yticks)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)

    ax.legend(loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=1, frameon=False,
                 borderpad=0, labelspacing=labelspacing, handlelength=0.5, columnspacing=1,
                 handletextpad=handletextpad, borderaxespad=0.6, markerscale=3)

def plot_fraction_significant_rules(prop, figsize=(1,1), dpi=600, x_lim=None, x_ticks=None, x_ticklabels=None,
                                   x_label=None, y_lim=None, y_ticks=None, y_ticklabels=None, y_label=None,
                                   axwidth=1.5, linewidth=1.5, xticks_pad=5, yticks_pad=0, labelpad_xlabel=5,
                                   labelpad_ylabel=0, color='black', fontsize=10, font='Arial'):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(prop, linewidth=linewidth, color=color, clip_on=False)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)

    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize, fontname=font, labelpad = labelpad_ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)

    ax.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    plt.show()

def plot_raster(sts, neuron_indices, t_lim=None, fontsize=10, color="black", x_label="", y_label="", markersize=0.05, 
figsize=(20, 3), font="arial", ax=None, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, tickwidth=2,
axwidth=3, dpi=200, ylabel_xloc=-0.1, ylabel_yloc=0.15, xlabel_xloc=0.4, xlabel_yloc=-0.1, x_milestones=None, linewidth_milestones=1.5, alpha_milestone=1, zorder_milestone=1):
    n_to_plot = len(neuron_indices)
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if x_milestones is not None:
        for x in x_milestones:
            ax.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones,alpha=alpha_milestone, zorder=zorder_milestone)
    ct = 0
    for neuron_num in neuron_indices:
        ax.scatter(sts[str(neuron_num)], np.full(len(sts[str(neuron_num)]), ct), linewidths=0, color=color, s=markersize, edgecolors=None, marker='o')
        ct += 1
    if t_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
    if (t_lim is None) and (x_ticks is not None):
        ax.set_xlim([x_ticks[0], x_ticks[-1]])
    if x_ticks is None:
        ax.set_xticks([t_lim[0], t_lim[1]])
    else:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)

    if x_label is not None:
        ax.plot((0.45, 0.55), (-0.05, -0.05), transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(xlabel_xloc, xlabel_yloc, x_label, fontsize=fontsize, fontname=font, ha='center')

    if y_label is not None:
        ax.plot((-0.05, -0.05), (0.45, 0.55), transform=ax.transAxes, color="black", clip_on=False, linewidth=axwidth)
        fig.text(ylabel_xloc, ylabel_yloc, y_label, fontsize=fontsize, fontname=font, rotation=90, ha='center')

    ax.set_ylim([-1, n_to_plot + 0.1])
    if y_ticks is None:
        ax.set_yticks([0, n_to_plot])
    else:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=tickwidth, labelsize=fontsize, length=tickwidth*2, pad = 10)
    ax.tick_params(axis='y', pad = 0)
    return(ax)

def plot_weights(w1=None, w2=None, ts=None, t_start=None, t_stop=None,
                axwidth=1.5, linewidth=1.5, n_to_plot_weights=100,
                fontsize=10, figsize=(1, 1), font = "arial", x_milestones=None, linewidth_milestones=1.5, alpha_milestone=1,
                color1=None, color2=None, label1=None, label2=None, ylog=False,
                x_ticks=None, x_ticklabels=None, x_label=None, alpha_w=None,  xlim=None,
                linewidth_weights=0.5,
                y_ticks=None, y_ticklabels=None, y_lim=None,
                dpi=600):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if ylog:
        ax.set_yscale('log')
    for syn_num in range(n_to_plot_weights):
        ax.plot(ts, w1[syn_num, :], color=color1, linewidth=linewidth_weights, alpha=alpha_w)
        ax.plot(ts, w2[syn_num, :], color=color2, linewidth=linewidth_weights, alpha=alpha_w)
    a4, =ax.plot(ts, np.mean(w1, axis=0), label=label1, color=color1, linewidth=linewidth, alpha=1)
    a3, =ax.plot(ts, np.mean(w2, axis=0), label=label2, color=color2, linewidth=linewidth, alpha=1)
    if x_milestones is not None:
        for x in x_milestones:
            ax.axvline(x=x, ymin=0, ymax=1, linestyle=(0, (2, 1)), color="black", linewidth=linewidth_milestones, zorder=1, alpha=alpha_milestone)
    if xlim is None:
        ax.set_xlim([t_start, t_stop])
    else:
        ax.set_xlim(xlim)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth, pad=0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=fontsize, fontname=font)
    ax.set_yticklabels(y_ticklabels, fontsize=fontsize, fontname=font)
    ax.tick_params(axis='x', which='minor', length=5*linewidth, color='black', width=linewidth, labelsize=fontsize, pad=10)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    leg = ax.legend(handles=[a3, a4], loc='upper center', bbox_to_anchor=(1.5, 1.5), fontsize=fontsize, ncol=1, frameon=False,
                borderpad=0, labelspacing=0.1, handlelength=0.1, columnspacing=0.5, handletextpad=0.1)
    plt.show()

def plot_2Dhist_contour(list_data1, list_data2,
                xlabel=None,
                ylabel=None,
                n_bins=100,
                range_2Dhist=[[-0.5, 1.5], [-0.5, 1.5]],
                font = "Arial", 
                fontsize = 10, 
                linewidth = 1.5, 
                xlim = None, 
                ylim = None,
                figsize=(1.5, 1),
                xticks=None,
                yticks=None,
                xhandlepad=None,
                yhandlepad=None,
                s=1,
                center_axes = False,
                dpi=600,
                color='black',
                color_xlabel="black",
                color_ylabel="black",
                marker='o',
                linewidth_line=1.5,
                color_line="black",
                alpha_line=0.5,
                xticklabels=None,
                yticklabels=None,
                zorder_line=0.1,
                colors=None,
                axwidth=1.5,
                bbox_to_anchor=(0.98, 0.9),
                level=3,
                gaussian_filt_sigma=0.71,
                labels_data=None):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n_data = len(list_data1)
    for i in range(n_data):
        H, xedges, yedges = np.histogram2d(list_data1[i], list_data2[i], bins=n_bins, range=range_2Dhist, density=True, weights=None)
        
        ax.contour(xedges[:-1], yedges[:-1], gaussian_filter(H.T, gaussian_filt_sigma), [level], 
                   colors=[colors[i]], linewidths=linewidth)
    ax.plot(range_2Dhist[0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
    ax.plot(range_2Dhist[0], [0,0], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)
    ax.plot([0,0], range_2Dhist[1], linestyle="--",linewidth=linewidth_line, color=color_line, alpha=alpha_line,zorder=zorder_line)


    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticks != None:
        ax.set_yticks(yticks)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)
    if center_axes:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.label.set_color(color_xlabel)
    ax.yaxis.label.set_color(color_ylabel)

    proxy = [plt.Rectangle((0,0),1,1,fc = color) for color in colors] 

    legend = ax.legend(proxy, labels_data, loc='upper left', bbox_to_anchor=bbox_to_anchor, 
              fontsize=fontsize, ncol=1, frameon=False,
              borderpad=0, labelspacing=0.5, handlelength=1, 
              columnspacing=0, handletextpad=0.3, borderaxespad=0.6)
    
    plt.show()

### Successor representation task

def compute_succ(n_tot=None, n_bins=None, test_starts=None, ontime_test=None, offtime_test=None,
                 n_rules=None, n_tests=None, n_fam=None, eng_rate=None, n_engs=None, method="both",
                 raw_r_rest=None, raw_r_pop=None):
    """
    Computes the successor representation metric on the sequential task
    We have n_rules rules simulated, with n_tests different test sessions (with increasing break time between train and test).
    before the task, we show each stimulus once to the network without plasticity => we define engram neurons for each stimulus.
    Task has 5 familiar stimuli and 2 novel stimuli => 7 engrams

    Inputs:
        eng_rates: [n_rules, n_tests, n_engrams, n_timepoints]
                typically data_from_computemetrics['eng_rate']
        n_bins: number of bins recorded for each test session, usually each bin is 0.1s
        test_starts: int, in number of bins, how long do we record before the testing actually starts (usually 1s sampled at 0.1s -> 10)
        ontime_test: int, in number of bins, how long is a stimulus active during test session
        offtime_test: int, in number of bins, how long do we wait between 2 stimuli being active
        n_rules: int, number of rules in the data in eng_rates
        n_tests: int, number of test sessions in the data in eng_rates
        n_fam: int, number of familiar stimuli
        n_eng: number of engrams, n_fam + n_nov (=n_tot elsewhere)
        method: string, which method to use for computing the succ metrics, "both" or "nov2". default both, nov2 only normalizes by the firing rate of the nov2 stimulus
        raw_r_rest: [n_rules, n_tests, n_timepoints]
            typically data_from_computemetrics['non_eng_rate'], only used if using method="rest"
    
    Outputs:
        succ: [n_rules, n_tests, 5, n_fam] 
            For each test session of each rule, compute 5 metrics: m-2, m-1, m0, m+1, m+2.
            When presenting one stimulus, how much is the engram of the +x stimulus in the sequence activated (done for each of the familiar stimulus in the test session)
            See exact definition in paper.
    """
    inds_single_stim = get_inds_single_stim_pres(n_tot, n_bins, test_starts, ontime_test, offtime_test)
    succ = np.zeros((n_rules, n_tests, 5, n_fam))
    for test_num in range(n_tests):
        resp_eng_stim = get_responses_engram_each_stim(eng_rate, inds_single_stim, test_num, n_rules, n_engs)
        if method == "rest":
            succ[:, test_num, :, :] = compute_succ_metric_1test(resp_eng_stim,
                                            method=method,
                                            raw_r_rest=raw_r_rest,
                                            inds_single_stim=inds_single_stim,
                                            test_num=test_num,
                                            n_rules=n_rules,
                                            n_engs=n_engs)
        elif method == "pop":
            succ[:, test_num, :, :] = compute_succ_metric_1test(resp_eng_stim,
                                            method=method,
                                            raw_r_pop=raw_r_pop,
                                            inds_single_stim=inds_single_stim,
                                            test_num=test_num,
                                            n_rules=n_rules,
                                            n_engs=n_engs)
        else:
            succ[:, test_num, :, :] = compute_succ_metric_1test(resp_eng_stim, method=method)
    return(succ)

def get_inds_single_stim_pres(n_tot, n_bins, test_starts, ontime_test, offtime_test):
    """
    Get the bins at which each stimulus is presented within the sequential task, first part of testing (no sequences).
    Usually called in compute_succ, see there for relevant variables
    """
    inds_single_stim = np.zeros( (n_tot, n_bins), dtype=bool )
    for i in range(n_tot):
        start = test_starts + i*(ontime_test + offtime_test) # removed -1 18/09/2024, didnt line up with pop rates going up.
        stop = start + ontime_test
        while start < stop:
            inds_single_stim[i,start]=1
            start += 1
    return(inds_single_stim)

def compute_succ_metric_1test(resp_eng_stim, method="both", raw_r_rest=None,
                              inds_single_stim=None, test_num=None, 
                              n_rules=None, n_engs=None, raw_r_pop=None):
    """
    Inputs:
        resp_eng_stim: [n_rules, n_engrams (engram number), n_engram (stimulus presented number)] output of get_responses_engram_each_stim()
    
    Outputs:
        succ_metrics: [n_rules, 5, n_fam]
    """
    r_lagged, r_novs = get_lagged_rates_1test(resp_eng_stim) #[n_rules, 5, n_fam] and [n_rules, 2, n_fam]
    succ_metric = np.zeros( (resp_eng_stim.shape[0], 5, 5) ) 
    if method == "both":
        aux_r_novs = np.mean(r_novs, axis=1)
        aux_r_novs = np.reshape(aux_r_novs, (resp_eng_stim.shape[0], 1, 5))
        aux_r_novs = np.repeat(aux_r_novs, 5, axis=1)
        succ_metric = r_lagged/(aux_r_novs+0.01)
    elif method == "nov2":
        aux_r_novs = np.reshape(r_novs[:,1,:], (resp_eng_stim.shape[0], 1, 5))
        aux_r_novs = np.repeat(aux_r_novs, 5, axis=1)
        succ_metric = r_lagged/(aux_r_novs+0.01)
    elif method == "rest":
        r_rest = get_r_rest_1test(inds_single_stim, raw_r_rest, n_rules, n_engs, test_num)
        aux_r_rest = np.repeat(r_rest, 5, axis=1)
        succ_metric = r_lagged/(aux_r_rest[:,:,2:]+0.01)
    elif method == "pop":
        r_pop = get_r_rest_1test(inds_single_stim, raw_r_pop, n_rules, n_engs, test_num)
        aux_r_pop = np.repeat(r_pop, 5, axis=1)
        succ_metric = r_lagged/(aux_r_pop[:,:,2:]+0.01)
    else:
        print("unknown method in compute_succ_metric_1test")
    return succ_metric

def get_lagged_rates_1test(resp_eng_stim):
        '''
        Part of sequential task, successor representation computation
        Assuming 5 familiar and 2 novel stimuli. 2 novel presented first then 5 familiar

        Inputs:
            resp_eng_stim: [n_rules, n_engrams (engram number), n_engram (stimulus presented number)] output of get_responses_engram_each_stim()
        
        Outputs:
            r_lagged [n_rules, 5, n_fam]: the engram rate of the stimulus -2 ago, -1, 0, +1, +2 in the sequence (5), computed at each familiar stimulus presentation (n_fam)
                e.g. +1: r_engfam[i+1]_during_fam[i]stim) (shape n_fam_stim, if i+1 > n_fam then cycle back to 0)
            r_novs [n_rules, 2, n_fam]: for each presented familiar stimulus, firing rate of the 2 engrams for the 2 novel stimuli (2), computed at each familiar stimulus presentation (n_fam)
        '''
        stim_label=['nov1','nov2','fam1','fam2','fam3','fam4','fam5']
        r_dict = dict()
        for i in range(7):
            r_dict[stim_label[i]] = dict()
            for j in range(7):
                r_dict[stim_label[i]]['during_'+stim_label[j]] = resp_eng_stim[:,i,j]
        r_lagged = np.zeros( (resp_eng_stim.shape[0], 5, 5) )
        r_novs = np.zeros( (resp_eng_stim.shape[0], 2, 5) )
        
        #-2 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam4' # during fam1 presentation look at eng_fam4 (-2)
        r_lagged[:,0,0] = r_dict[label_engram][label_stim_pres]
        r_novs[:,0,0] = r_dict['nov1'][label_stim_pres]
        r_novs[:,1,0] = r_dict['nov2'][label_stim_pres]
        label_stim_pres = 'during_fam2'; label_engram = 'fam5'
        r_lagged[:,0,1] = r_dict[label_engram][label_stim_pres]
        r_novs[:,0,1] = r_dict['nov1'][label_stim_pres]
        r_novs[:,1,1] = r_dict['nov2'][label_stim_pres]
        label_stim_pres = 'during_fam3'; label_engram = 'fam1'
        r_lagged[:,0,2] = r_dict[label_engram][label_stim_pres]
        r_novs[:,0,2] = r_dict['nov1'][label_stim_pres]
        r_novs[:,1,2] = r_dict['nov2'][label_stim_pres]
        label_stim_pres = 'during_fam4'; label_engram = 'fam2'
        r_lagged[:,0,3] = r_dict[label_engram][label_stim_pres]
        r_novs[:,0,3] = r_dict['nov1'][label_stim_pres]
        r_novs[:,1,3] = r_dict['nov2'][label_stim_pres]
        label_stim_pres = 'during_fam5'; label_engram = 'fam3'
        r_lagged[:,0,4] = r_dict[label_engram][label_stim_pres]
        r_novs[:,0,4] = r_dict['nov1'][label_stim_pres]
        r_novs[:,1,4] = r_dict['nov2'][label_stim_pres]
        
        #-1 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam5' # during fam1 presentation look at eng_fam5 (-1)
        r_lagged[:,1,0] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam2'; label_engram = 'fam1'
        r_lagged[:,1,1] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam3'; label_engram = 'fam2'
        r_lagged[:,1,2] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam4'; label_engram = 'fam3'
        r_lagged[:,1,3] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam5'; label_engram = 'fam4'
        r_lagged[:,1,4] = r_dict[label_engram][label_stim_pres]
        
        #0 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam1' # during fam1 presentation look at eng_fam1 (0)
        r_lagged[:,2,0] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam2'; label_engram = 'fam2'
        r_lagged[:,2,1] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam3'; label_engram = 'fam3'
        r_lagged[:,2,2] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam4'; label_engram = 'fam4'
        r_lagged[:,2,3] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam5'; label_engram = 'fam5'
        r_lagged[:,2,4] = r_dict[label_engram][label_stim_pres]
        
        #+1 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam2' # during fam1 presentation look at eng_fam2 (+1)
        r_lagged[:,3,0] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam2'; label_engram = 'fam3'
        r_lagged[:,3,1] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam3'; label_engram = 'fam4'
        r_lagged[:,3,2] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam4'; label_engram = 'fam5'
        r_lagged[:,3,3] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam5'; label_engram = 'fam1'
        r_lagged[:,3,4] = r_dict[label_engram][label_stim_pres]
        
        #+2 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam3' # during fam1 presentation look at eng_fam3 (+2)
        r_lagged[:,4,0] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam2'; label_engram = 'fam4'
        r_lagged[:,4,1] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam3'; label_engram = 'fam5'
        r_lagged[:,4,2] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam4'; label_engram = 'fam1'
        r_lagged[:,4,3] = r_dict[label_engram][label_stim_pres]
        label_stim_pres = 'during_fam5'; label_engram = 'fam2'
        r_lagged[:,4,4] = r_dict[label_engram][label_stim_pres]
        
        return(r_lagged, r_novs)

def get_responses_engram_each_stim(eng_rates, inds_single_stim, test_num, n_rules, n_engs):
        '''
        Get the activity of each engram during each stimulus presentation
        Part of the sequential task and successor reprensation metric, typically called by compute_succ.

        Inputs:
            eng_rates: [n_rules, n_tests, n_engrams, n_timepoints]
                see compute_succ
            inds_single_stim: [n_engrams, n_timepoints]
                which timepoints correspond to presentation of each stimulus:  
                inds_stim[0] = [0,0,0,1,1,0] means that the first engram was presented at the 4th and 5th timepoints
                typically output of get_inds_single_stim_pres()

        Returns: 
            r_engrams [n_rules, n_engrams (engram number), n_engram (stimulus presented number)]:  
                activity of each engram during each stimulus presentation
                r_engrams[a][b] is mean activity of the neurons corresponding to engram "a" while stimulus "b" is presented.
                stimulus ordering assumed is from pretrainingeng (not from testing in case they are different)
        '''
        r_engrams = np.zeros( (n_rules, n_engs, n_engs) )
        for engram_num in range(n_engs):
            for presented_stim_num in range(n_engs):
                r_engrams[:,engram_num,presented_stim_num] = np.mean(eng_rates[:,test_num,engram_num,inds_single_stim[presented_stim_num]], axis=1)
        return(r_engrams)

def get_r_rest_1test(inds_single_stim, raw_r_rest, n_rules, n_engs, test_num):
        '''
        Part of sequential task, successor representation computation
        Assuming 5 familiar and 2 novel stimuli. 2 novel presented first then 5 familiar

        Inputs:
            inds_single_stim: [n_tot, n_bins] when is each stimulus presented during a test session
                                output of get_inds_single_stim_pres()
            raw_r_rest: [n_rules, n_tests, n_bins] typically the "non_eng_rate" or "pop_rate" field of a metric file.
        
        Outputs:
            r_rest [n_rules, 1, n_tot]: for each presented familiar stimulus, firing rate of pop of neuron that are in no familiar engrams (1), computed at each familiar stimulus presentation (n_fam)
        '''
        r_rest = np.zeros( (n_rules, 1, n_engs) )
        for presented_stim_num in range(n_engs):
            r_rest[:,0,presented_stim_num] = np.mean(raw_r_rest[:,test_num,inds_single_stim[presented_stim_num]], axis=1)
        return(r_rest)

def get_rsucc_rnov(n_tot=None, n_bins=None, test_starts=None, ontime_test=None, offtime_test=None,
                 n_rules=None, n_tests=None, n_fam=None, eng_rate=None, n_engs=None):
    inds_single_stim = get_inds_single_stim_pres(n_tot, n_bins, test_starts, ontime_test, offtime_test)
    fams = np.zeros((n_rules, n_tests, 5, n_fam))
    novs = np.zeros((n_rules, n_tests, 2, n_fam))
    for test_num in range(n_tests):
        resp_eng_stim = get_responses_engram_each_stim(eng_rate, inds_single_stim, test_num, n_rules, n_engs)
        fams[:, test_num, :, :], novs[:, test_num, :, :] = get_lagged_rates_1test(resp_eng_stim)
    return(fams, novs)

def get_r_rest(n_tot, n_bins, test_starts, ontime_test, offtime_test, raw_r_rest, n_rules, n_engs, n_tests):
        '''
        Part of sequential task, successor representation computation
        Assuming 5 familiar and 2 novel stimuli. 2 novel presented first then 5 familiar

        Inputs:
            inds_single_stim: [n_tot, n_bins] when is each stimulus presented during a test session
                                output of get_inds_single_stim_pres()
            raw_r_rest: [n_rules, n_tests, n_bins] typically the "non_eng_rate" or "pop_rate" field of a metric file.
        
        Outputs:
            r_rest [n_rules, n_tests, 1, n_tot]: for each presented familiar stimulus, firing rate of pop of neuron that are in no familiar engrams (1), computed at each familiar stimulus presentation (n_fam)
        '''
        inds_single_stim = get_inds_single_stim_pres(n_tot, n_bins, test_starts, ontime_test, offtime_test)
        r_rest = np.zeros( (n_rules, n_tests, 1, n_engs) )
        for test_num in range(n_tests):
            for presented_stim_num in range(n_engs):
                r_rest[:,test_num,0,presented_stim_num] = np.mean(raw_r_rest[:,test_num,inds_single_stim[presented_stim_num]], axis=1)
        return(r_rest)


### Contextual novelty task

def get_dr_stfam(n_bins=None,
            test_starts=None,
            ontime_test=None,
            offtime_test=None,
            rpop=None,
            n_fam=None,
            n_nov=None):

    r_famfam, r_famnov, r_novnov = get_stfam(n_bins=n_bins, test_starts=test_starts, ontime_test=ontime_test, offtime_test=offtime_test,
                                             rpop=rpop, n_fam=n_fam, n_nov=n_nov)
    dr_novnov = 2*(r_novnov - r_famfam) / (r_novnov + r_famfam+0.01)
    dr_famnov = 2*(r_famnov - r_famfam) / (r_famnov + r_famfam+0.01)
    return(dr_famnov, dr_novnov)

def get_stfam(n_bins=None,
            test_starts=None,
            ontime_test=None,
            offtime_test=None,
            rpop=None,
            n_fam=None,
            n_nov=None):
    """
    TODO
    """

    inds_stim = get_inds_last_el_sequence_stim_pres(n_bins, test_starts, ontime_test, offtime_test,n_fam,n_nov)
    r_famfam =  np.mean( [np.mean(rpop[:,:,inds_stim[i]], axis=2) for i in range(5)], axis=0 )
    r_famnov =  np.mean( [np.mean(rpop[:,:,inds_stim[i]], axis=2) for i in range(5,10)], axis=0 )
    r_novnov =  np.mean( [np.mean(rpop[:,:,inds_stim[i]], axis=2) for i in range(10,15)], axis=0 )
    return(r_famfam, r_famnov, r_novnov) 

def get_inds_last_el_sequence_stim_pres(n_bins, test_starts, ontime_test, offtime_test, n_fam, n_nov):
    n_seqs = 3*n_fam #sfamtfam, sfamtnov and snovtnov
    inds_seq_stim = np.zeros( (n_seqs, n_bins), dtype=bool )
    for i in range(n_seqs):
        start = test_starts + (n_fam+n_nov)*(ontime_test + offtime_test) +\
        + i*(4*ontime_test + offtime_test) + 3*ontime_test
        stop = start + ontime_test
        while start < stop:
            inds_seq_stim[i,start]=1
            start += 1
    return(inds_seq_stim)   


####### Replay analysis

def get_rhos_1tbreak_1rule(eng_rates, rule_ind, break_ind, n_fam=5, n_nov=2):
    nov_rates_av = np.mean(eng_rates[rule_ind, break_ind, :n_nov, :], axis=0) + 0.1
    return(np.array([eng_rates[rule_ind, break_ind, n_nov + (i+1)%n_fam, :]/nov_rates_av for i in range(n_fam)]))

def get_t_i(rho_i, ind_t_start, ind_t_end):
    return(np.argmax(rho_i[ind_t_start:ind_t_end]) + ind_t_start)

def get_rhos(eng_rates, n_fam=5, n_nov=2):
    nov_rates_av = np.mean(eng_rates[:, :, :n_nov, :], axis=2) + 0.1
    # print(nov_rates_av.shape)
    return(np.array([eng_rates[:, :, n_nov + (i+1)%n_fam, :]/nov_rates_av for i in range(n_fam)]))

def get_inds_start_single_stim_pres(test_starts, ontime_test, offtime_test, n_fam=5):
    start_inds = np.zeros(n_fam, dtype=int)
    stop_inds = np.zeros(n_fam, dtype=int)
    for i in range(n_fam): 
        start_inds[i] = test_starts + (i+2)*(ontime_test + offtime_test) # first 2 stim are novel
        stop_inds[i] = start_inds[i] + ontime_test + offtime_test - 1
    return(start_inds, stop_inds)

def get_ts_maxrhos(rhos, inds_start, inds_stop, n_fam=5):
    n_rules = rhos.shape[1]
    n_breaks = rhos.shape[2]
    ts_max = np.zeros( (n_fam, n_rules, n_breaks), dtype=int)
    for i in range(n_fam):
        ts_max[i] = np.argmax(rhos[i, :, :, inds_start[i]:inds_stop[i]], axis=2) + inds_start[i] # we want to look at response of stim i while we showed stim i-1 to the net, that lag is already in rho
    return(ts_max)

def plot_replay_metrics_1rule(rhos_max_av, rhos_spec_av, ts_max, axwidth=1.5, linewidth=1.5, fontsize=10, xticks_pad=1, yticks_pad=0, font='arial'):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(1, 1.5), dpi=300)

    ax1.plot(rhos_max_av, 'black', linewidth=linewidth)
    ax1.set_ylabel(r'$\rho_{max}^{rep}$', fontname=font, fontsize=fontsize, labelpad=1)
    ax1.set_xlim([0,9])
    ax1.set_xticks([0,5,9], labels=["", "", ""])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(axwidth)
    ax1.spines['left'].set_linewidth(axwidth)
    ax1.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax1.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    ax2.plot(rhos_spec_av, 'blue', linewidth=linewidth)
    ax2.set_ylabel(r'$\rho_{spec}^{rep}$', fontname=font, fontsize=fontsize, labelpad=1)
    ax2.set_xlim([0,9])
    ax2.set_xticks([0,5,9], labels=["", "", ""])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(axwidth)
    ax2.spines['left'].set_linewidth(axwidth)
    ax2.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax2.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    ax3.plot(ts_max, 'green', linewidth=linewidth)
    ax3.set_ylabel(r'$t_{i}^{*}$', fontname=font, fontsize=fontsize, labelpad=1)
    ax3.set_xlim([0,9])
    ax3.set_xticks([0,5,9], labels=["1s", "5min", "4h"])
    ax3.set_ylim([0,30])
    ax3.set_yticks([0,30])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(axwidth)
    ax3.spines['left'].set_linewidth(axwidth)
    ax3.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax3.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    
    plt.show()


### Spiking network analysis, computing engrams

def get_eng_rate(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests, n_recorded, n_tot_stim, engrams):
    binned_spikes_big = get_binned_spikes_big(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests, n_recorded)
    eng_rate = np.zeros((n_tests, n_tot_stim, binned_spikes_big.shape[2]))
    for engram_num in range(n_tot_stim):
        neur_ind_keep = [True if i in engrams[engram_num] else False for i in range(n_recorded)]
        eng_rate[:,engram_num, :] = np.mean(binned_spikes_big[:,neur_ind_keep,:], axis=1)
    return(eng_rate/bin_size_big)

def get_binned_spikes_big(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests,
                          n_recorded):
    """
    computes binned_spikes for each recorded neuron. 
    Since we have several test sessions during a single simulation, we use a type
    ts = np.array[n_test_phases, n_bins]
    binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
    """
    n_bins_per_test = len( np.arange( test_starts[0]-l_pre_test_record-0.001 ,  test_starts[0]+l_1test+.001 , bin_size_big ) ) - 1
    ts_big = np.zeros((n_tests, n_bins_per_test))
    binned_spikes_big = np.zeros((n_tests, n_recorded, n_bins_per_test))
    for i in range(n_tests):
        bins = np.arange(test_starts[i]-l_pre_test_record-0.001, test_starts[i]+l_1test+0.001, bin_size_big)
        binned_spikes_big[i] = np.array([np.histogram(spiketimes[str(neuron_num)], bins=bins)[0] \
                for neuron_num in range(n_recorded)])
        ts_big[i] = bins[:-1]
    return(binned_spikes_big)

def get_engram_neurons(spiketimes, start_times, duration=1, n_tot_stim=7, frac_size_engram=0.1, n_neurons=4096):
    n_engrams = n_tot_stim
    n_neurons_per_engrams = int(frac_size_engram*n_neurons)
    engrams = np.zeros((n_engrams,n_neurons_per_engrams), dtype=int)

    #for each stimulus, pick the top N most active neurons (given by frac_size)
    for stim_num in range(n_tot_stim):
        #get individual firing rates
        rs = get_individual_rates(spiketimes, start_times[stim_num], 
                                        start_times[stim_num] + duration)
        sorted_array = np.argsort(rs)
        engrams[stim_num] = sorted_array[-n_neurons_per_engrams:]

    engrams_dict = dict()
    for i in range(n_engrams):
        engrams_dict[str(i)] = engrams[i]
    return(engrams_dict)

def get_individual_rates(spiketimes, start, stop, n_neurons=4096):
        """
        Compute the the firing of each recorded (excitatory) neuron 
        between start and stop
        """
        rates = np.zeros(n_neurons)
        for neuron in range(n_neurons):
            rates[neuron] =  np.sum(np.logical_and(start<=spiketimes[str(neuron)], 
                                                   spiketimes[str(neuron)]<=stop))/(stop-start)
        return(rates)

def read_monitor_spiketime_files(filename, num_neurons=4096):

    spiketimes = {str(neuron): [] for neuron in range(num_neurons)}

    try:
        f = open(filename, "r")
    except:
        print("problem with st file, returning None")
        return(None)
    lines = f.readlines()
    for line in lines:
        aux = line.split(" ")
        try:
            spiketimes[str(int(aux[1]))].append(float(aux[0]))
        except:
            print("problem with file", filename)
    return spiketimes

def read_monitor_spiketime_file_nparray(filename, num_neurons=4096):

    spiketimes = {str(neuron): [] for neuron in range(num_neurons)}

    try:
        f = open(filename, "r")
    except:
        print("problem with st file, returning None")
        return(None)
    lines = f.readlines()
    for line in lines:
        aux = line.split(" ")
        try:
            spiketimes[str(int(aux[1]))].append(float(aux[0]))
        except:
            print("problem with file", filename)
    for neuron in range(num_neurons):
        spiketimes[str(neuron)] = np.array(spiketimes[str(neuron)])
    return spiketimes

def load_w_mat(filename):
    """
    loads a full weight matrix from file as a dictionnary. weight matrix itself stored as a structured array
    Format: np.array([int pre, int post, float weight])
    Careful: Matrix market format starts at 1
    w["n_from"] int
    w["n_to"] int
    w["n_w_non_zero"] int
    w["w"] array(dtype=('pre',np.uint16),('post',np.uint16),('w',np.float32))
    """    
    
    with open(filename) as file:
        for i, line in enumerate(file):
            aux = line.split(" ")
            if i >= 6:
                ws["w"][i-6] = (int(aux[0]), int(aux[1]), float(aux[2]))
            elif i < 5:
                pass
            elif i == 5:
                n_from = int(aux[0])
                n_to = int(aux[1])
                n_w_non_zero = int(aux[2])
                print("matrix found from", n_from, "to", n_to, "neurons.", np.round(n_w_non_zero/n_from/n_to*100, 2), "% sparsity")
                ws = dict()
                ws["n_from"]=n_from
                ws["n_to"]=n_to
                ws["n_w_non_zero"]=n_w_non_zero
                dt = np.dtype([('pre',np.uint16),('post',np.uint16),('w',np.float32)])
                ws["w"]=np.zeros( (ws["n_w_non_zero"],), dt )
            
    return(ws)

def get_mean_weight(w, pre_targets, post_targets):
    pre_w_ind = np.in1d(w["w"]['pre'], pre_targets)
    post_w_ind = np.in1d(w["w"]['post'], post_targets)
    pre_and_post_w_ind = np.logical_and(pre_w_ind, post_w_ind)
    return(np.mean(w["w"]['w'][pre_and_post_w_ind]))

def translate_engram(engram):
    translated_engram = dict()
    for key in engram.keys():
        translated_engram[key] = [i + 1 for i in engram[key]]
    return(translated_engram)

def get_non_engrams(list_stim_consider, engram_dict, n_neurons=4096):
    ind_engram_neurons = []
    for stim_num in list_stim_consider:
        ind_engram_neurons += list(engram_dict[str(stim_num)])
    ind_engram_neurons = np.unique(ind_engram_neurons)
    non_engram_neurons = []
    for neuron in range(n_neurons):
        if neuron not in ind_engram_neurons:
            non_engram_neurons.append(neuron)
    return(np.array(non_engram_neurons))

def get_eng_rate_npy(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests, n_recorded, n_tot_stim, engrams):
    binned_spikes_big = get_binned_spikes_big_npy(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests, n_recorded)
    eng_rate = np.zeros((n_tests, n_tot_stim, binned_spikes_big.shape[2]))
    for engram_num in range(n_tot_stim):
        neur_ind_keep = [True if i in engrams[engram_num] else False for i in range(n_recorded)]
        eng_rate[:,engram_num, :] = np.mean(binned_spikes_big[:,neur_ind_keep,:], axis=1)
    return(eng_rate/bin_size_big)

def get_binned_spikes_big_npy(spiketimes, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests,
                          n_recorded):
    """
    computes binned_spikes for each recorded neuron. 
    Since we have several test sessions during a single simulation, we use a type
    ts = np.array[n_test_phases, n_bins]
    binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
    """
    n_bins_per_test = len( np.arange( test_starts[0]-l_pre_test_record-0.001 ,  test_starts[0]+l_1test+.001 , bin_size_big ) ) - 1
    ts_big = np.zeros((n_tests, n_bins_per_test))
    binned_spikes_big = np.zeros((n_tests, n_recorded, n_bins_per_test))
    for i in range(n_tests):
        bins = np.arange(test_starts[i]-l_pre_test_record-0.001, test_starts[i]+l_1test+0.001, bin_size_big)
        binned_spikes_big[i] = np.array([np.histogram(spiketimes[str(neuron_num)], bins=bins)[0] \
                for neuron_num in range(n_recorded)])
        ts_big[i] = bins[:-1]
    return(binned_spikes_big)

def get_engram_neurons_npy(n_tot_stim, lpt, l_stim_on_pretraineng, l_stim_off_pretraineng, frac_size_engram, n_recorded, spiketimes):
    #goal: get the top frac_size most active neurons during presentation 
    #of each stimulus before we even start the task

    #get the start times of each stimulus during pretrain_assessment
    start_times_stim_pretraineng = np.zeros(n_tot_stim)
    start_times_stim_pretraineng[0] = lpt
    for i in range(1,n_tot_stim):
        start_times_stim_pretraineng[i] = start_times_stim_pretraineng[i-1] + l_stim_on_pretraineng + l_stim_off_pretraineng

    n_engrams = n_tot_stim
    n_neurons_per_engrams = int(frac_size_engram*n_recorded)
    engrams = np.zeros((n_engrams,n_neurons_per_engrams), dtype=int)

    #for each stimulus, pick the top N most active neurons (given by frac_size)
    for stim_num in range(n_tot_stim):
        #get individual firing rates
        rs = get_individual_rates_npy(start_times_stim_pretraineng[stim_num], 
                                start_times_stim_pretraineng[stim_num] + l_stim_on_pretraineng,
                                spiketimes, n_recorded)
        sorted_array = np.argsort(rs)
        engrams[stim_num] = sorted_array[-n_neurons_per_engrams:]
    return(engrams)

def get_individual_rates_npy(start, stop, spiketimes, n_recorded):
    """
    Compute the the firing of each recorded (excitatory) neuron 
    between start and stop
    """
    rates = np.zeros(n_recorded)
    for neuron in range(n_recorded):
        rates[neuron] =  np.sum(np.logical_and(start<=spiketimes[str(neuron)], 
                                            spiketimes[str(neuron)]<=stop))/(stop-start)
    return(rates)


### Allen data analysis

def get_dr_perAnimal_1region(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_keys = len(keys)
        mdr_perAnimal = np.zeros(n_keys)
        print('number of sessions (animals):', n_keys)
        for i, k in enumerate(keys):
            mean_fam_1region_per_cell = np.nanmean(f[k]['familiar'], axis=0)
            mean_nov_1region_per_cell = np.nanmean(f[k]['novel'], axis=0)
            dr_1region_per_cell = 2*(mean_nov_1region_per_cell - mean_fam_1region_per_cell) / (mean_nov_1region_per_cell + mean_fam_1region_per_cell+0.01)
            mdr_perAnimal[i] = np.mean(dr_1region_per_cell)
    return mdr_perAnimal


### Synaptic data comparison

def get_dw(rule_params, dt_values):
    tau_pre = rule_params[0]
    tau_post = rule_params[1]
    alpha = rule_params[2]
    beta = rule_params[3]
    gamma = rule_params[4]
    kappa = rule_params[5]
    dw_values = np.zeros(len(dt_values))
    for dt in range(len(dt_values)):
        if dt_values[dt] > 0:
            dw_values[dt] = alpha + beta + gamma * np.exp(-dt_values[dt]/tau_pre)
        else:
            dw_values[dt] = alpha + beta + kappa * np.exp(dt_values[dt]/tau_post)
    return dw_values

def plot_4_rules_wData(thetas,
                n_bins=1000,
                x_lim=[-0.2,0.2],
                y_lim = [-1,1],
                x_datapoints = [],
                y_datapoints = [],
                ind_plot_datapoint = 0,
                markersize_datapoints = 1,
                linewidth_data_marker=0.4,
                y_ticks=[],
                x_ticks=[],
                x_ticklabels=None,
                x_label=r'$\Delta t$',
                y_label=r'$\Delta w$',
                color_ee=(165/256,42/256,42/256),
                color_ei=(242/256, 140/256, 40/256),
                color_ie=(8/256, 143/256, 143/256),
                color_ii=(47/256, 85/256, 151/256),
                color_ylabel="black",
                figsize=(0.6,0.1),
                labelpad_xlabel=1,
                fontsize=10,
                labelpad_ylabel=27,
                linewidth=0.8,
                axwidth=0.8,
                dpi=600,
                xticks_pad=0,
                yticks_pad=0,
                rotation=0,
                font='arial',
                y_ticklabels=None,
                color_data='black'):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    ts = np.linspace(x_lim[0], x_lim[1],num=n_bins)
    ind_t_pos = 0
    while ts[ind_t_pos] < 0:
        ind_t_pos += 1

    for theta_num in range(len(thetas)):
        dws = np.array([thetas[theta_num, 2] + thetas[theta_num, 3] + thetas[theta_num, 5]*np.exp(-np.abs(ts[i])/thetas[theta_num, 1]) for i in range(ind_t_pos)])
        dws = np.append(dws, np.array([thetas[theta_num, 2] + thetas[theta_num, 3] + thetas[theta_num, 4]*np.exp(-np.abs(ts[i])/thetas[theta_num, 0]) for i in range(ind_t_pos, len(ts))]), axis=0)
        dws = dws / np.max(np.abs(dws))
        line1_1, = ax1.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ee, linewidth=linewidth, clip_on = False, zorder=0.1)
        line1_2, = ax1.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ee, linewidth=linewidth, clip_on = False, zorder=0.1)
        line1_1.set_solid_capstyle('round')
        line1_2.set_solid_capstyle('round')
    ax1.set_xlim(x_lim)
    ax1.set_xticks(x_ticks)
    ax1.set_ylim([-1, 1])
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels(x_ticklabels)
    ax1.set_yticklabels(y_ticklabels)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_linewidth(axwidth)
    ax1.spines['left'].set_linewidth(axwidth)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['left'].set_position('zero')
    ax1.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax1.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    for theta_num in range(len(thetas)):
        dws = np.array([thetas[theta_num, 2+6] + thetas[theta_num, 3+6] + thetas[theta_num, 5+6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 1+6]) for i in range(ind_t_pos)])
        dws = np.append(dws, np.array([thetas[theta_num, 2+6] + thetas[theta_num, 3+6] + thetas[theta_num, 4+6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 0+6]) for i in range(ind_t_pos, len(ts))]), axis=0)
        dws = dws / np.max(np.abs(dws))
        line2_1, = ax2.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ei, linewidth=linewidth, clip_on = False, zorder=0.1)
        line2_2, = ax2.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ei, linewidth=linewidth, clip_on = False, zorder=0.1)
        line2_1.set_solid_capstyle('round')
        line2_2.set_solid_capstyle('round')
    ax2.set_xlim(x_lim)
    ax2.set_xticks(x_ticks)
    ax2.set_ylim([-1, 1])
    ax2.set_yticks(y_ticks)
    ax2.set_xticklabels(x_ticklabels)
    ax2.set_yticklabels(y_ticklabels)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_linewidth(axwidth)
    ax2.spines['left'].set_linewidth(axwidth)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['left'].set_position('zero')
    ax2.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax2.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)

    for theta_num in range(len(thetas)):
        dws = np.array([thetas[theta_num, 2+2*6] + thetas[theta_num, 3+2*6] + thetas[theta_num, 5+2*6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 1+2*6]) for i in range(ind_t_pos)])
        dws = np.append(dws, np.array([thetas[theta_num, 2+2*6] + thetas[theta_num, 3+2*6] + thetas[theta_num, 4+2*6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 0+2*6]) for i in range(ind_t_pos, len(ts))]), axis=0)
        dws = dws / np.max(np.abs(dws))
        line3_1, = ax3.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ie, linewidth=linewidth, clip_on = False, zorder=0.1)
        line3_2, = ax3.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ie, linewidth=linewidth, clip_on = False, zorder=0.1)
        line3_1.set_solid_capstyle('round')
        line3_2.set_solid_capstyle('round')
    ax3.set_xlim(x_lim)
    ax3.set_xticks(x_ticks)
    ax3.set_ylim([-1, 1])
    ax3.set_yticks(y_ticks)
    ax3.set_xticklabels(x_ticklabels,)
    ax3.set_yticklabels(y_ticklabels)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_linewidth(axwidth)
    ax3.spines['left'].set_linewidth(axwidth)
    ax3.spines['bottom'].set_position('zero')
    ax3.spines['left'].set_position('zero')
    ax3.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax3.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line3_1.set_solid_capstyle('round')
    line3_2.set_solid_capstyle('round')

    for theta_num in range(len(thetas)):
        dws = np.array([thetas[theta_num, 2+3*6] + thetas[theta_num, 3+3*6] + thetas[theta_num, 5+3*6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 1+3*6]) for i in range(ind_t_pos)])
        dws = np.append(dws, np.array([thetas[theta_num, 2+3*6] + thetas[theta_num, 3+3*6] + thetas[theta_num, 4+3*6]*np.exp(-np.abs(ts[i])/thetas[theta_num, 0+3*6]) for i in range(ind_t_pos, len(ts))]), axis=0)
        dws = dws / np.max(np.abs(dws))
        line4_1, = ax4.plot(ts[:n_bins//2], dws[:n_bins//2], color=color_ii, linewidth=linewidth, clip_on = False, zorder=0.1)
        line4_2, = ax4.plot(ts[n_bins//2:], dws[n_bins//2:], color=color_ii, linewidth=linewidth, clip_on = False, zorder=0.1)
        line4_1.set_solid_capstyle('round')
        line4_2.set_solid_capstyle('round')
    ax4.set_xlim(x_lim)
    ax4.set_xticks(x_ticks)
    ax4.set_ylim([-1, 1])
    ax4.set_yticks(y_ticks)
    ax4.set_xticklabels(x_ticklabels)
    ax4.set_yticklabels(y_ticklabels)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_linewidth(axwidth)
    ax4.spines['left'].set_linewidth(axwidth)
    ax4.spines['bottom'].set_position('zero')
    ax4.spines['left'].set_position('center')
    ax4.tick_params(axis='x', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=xticks_pad)
    ax4.tick_params(axis='y', width=axwidth, labelsize=fontsize, length=2*axwidth, pad=yticks_pad)
    line4_1.set_solid_capstyle('round')
    line4_2.set_solid_capstyle('round')

    if ind_plot_datapoint == 0:
        chosen_ax = ax1
    elif ind_plot_datapoint == 1:
        chosen_ax = ax2
    elif ind_plot_datapoint == 2:
        chosen_ax = ax3
    elif ind_plot_datapoint == 3:
        chosen_ax = ax4
    chosen_ax.scatter(x_datapoints, y_datapoints, color=color_data, s=markersize_datapoints, zorder=10, marker='.', edgecolor='black', linewidth=linewidth_data_marker, clip_on = False)
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.show()

def plot_distance_distribution(
        data_1D=None,
        dpi=600,
        n_bins=None, 
        range = None,
        log_scale=False,
        x_lim=None,
        x_ticks=None,
        x_ticklabels=None,
        labelpad_xlabel=1,
        x_label=None,
        y_lim=None,
        y_ticks=None,
        y_ticklabels=None,
        labelpad_ylabel=1,
        y_label=None, 
        figsize=(2,1),
        linewidth=1,
        title=None,
        fontsize=10,
        font="arial",
        color='black',
        rotation=45):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    hist, bins = np.histogram(data_1D, bins=n_bins, density=False, range=range)
    line1, = ax.plot(bins[1:], hist, color=color, linewidth=linewidth)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth)
    ax.minorticks_off()
    
    if log_scale:
        ax.set_yscale('log')
    if x_lim is not None:
        ax.set_xlim([x_lim[0], x_lim[1]])
        ax.set_xticks([x_lim[0], x_lim[1]])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels, rotation = rotation)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = labelpad_xlabel)
    if title is not None:
        ax.set_title(label=title, fontsize=fontsize*1.2)
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    if y_label is not None:
        ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = labelpad_ylabel)

    plt.show()

def plot_spider_meandiffs(
        mean1,
        mean2,
        labels = np.array([r'$\tau_{pre}$', r'$\tau_{post}$', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\kappa$']),
        color_fill = 'black',
        color_mean = 'blue',
        figsize = (0.8,0.8),
        axwidth = 1,
        linewidth=1,
        fontsize=10,
        start_draw = 1.5,
        y_lim = [-3,2]):
    
    # because polar plot, loop end on start
    stats_mean1 = np.concatenate((mean1,[mean1[0]]))
    stats_mean2 = np.concatenate((mean2,[mean2[0]]))

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, stats_mean1, '-', color=color_mean, linewidth=linewidth, alpha=1, zorder=11)
    ax.plot(angles, stats_mean2, '-', color=color_fill, linewidth=linewidth, alpha=1, zorder=10.5)
    ax.fill_between(angles, stats_mean2, stats_mean1, color=color_fill, alpha=1, zorder=10, linewidth=0)

    ax.grid(False)
    hex_radii = [0] #[-2,0,2]
    for r in hex_radii:
        ax.plot(
            angles,
            np.full_like(angles, r),
            color='black',
            linewidth=0.8,
            zorder=0,
            linestyle = (0, (2, 1)),
            alpha=1
        )

    for angle in angles[:-1]:
        ax.plot(
            [angle, angle],
            [start_draw, y_lim[1]],
            color='black',
            linewidth=axwidth,
            zorder=1,
        )
    
    ax.scatter([0], y_lim[0], color='black', s=1, zorder=20)

    ax.spines['polar'].set_visible(False)
    ax.set_thetagrids(angles * 180/np.pi, np.concatenate((labels,[labels[0]])))
    plt.yticks([0],[""],color="black", size=fontsize)
    plt.ylim(y_lim)
    plt.show()

def plot_4spider_plots(d, sorted_inds):
    for chosen_con in ['ee', 'ei', 'ie', 'ii']:
        if chosen_con == 'ee':
            color = color_ee
            start_ind = 0
            end_ind = 6
        elif chosen_con == 'ei':
            color = color_ei
            start_ind = 6
            end_ind = 12
        elif chosen_con == 'ie':
            color = color_ie
            start_ind = 12
            end_ind = 18
        elif chosen_con == 'ii':
            color = color_ii
            start_ind = 18
            end_ind = 24

        y_lim = [-1.6,1.8]

        all_rules = np.array(d['theta'][:, start_ind:end_ind])
        all_rules[:,0] = (all_rules[:,0]-0.01)/(0.1 - 0.01)*y_lim[1]
        all_rules[:,1] = (all_rules[:,1]-0.01)/(0.1 - 0.01)*y_lim[1]
        mean_all_rules = np.mean(all_rules, axis=0)
        std_all_rules = np.std(all_rules, axis=0)

        n_rules_to_analyse = 10
        rules_selected = np.array(d[sorted_inds[:n_rules_to_analyse]]['theta'][:,start_ind:end_ind])
        rules_selected[:,0] = (rules_selected[:,0]-0.01)/(0.1 - 0.01)*y_lim[1]
        rules_selected[:,1] = (rules_selected[:,1]-0.01)/(0.1 - 0.01)*y_lim[1]
        mean_rules_selected = np.mean(rules_selected, axis=0)
        std_rules_selected = np.std(rules_selected, axis=0)

        plot_spider_meandiffs(
            mean_all_rules,
            mean_rules_selected,
            color_fill = color,
            color_mean = color,
            figsize = (0.95,0.95),
            axwidth = 1,
            linewidth = 0.8,
            y_lim = y_lim,
            start_draw = 1.4,
        )

def plot_drmem_comparison(dr, n_good_rules, sorted_inds, marker_size_circ = 6, elinewidth_small = 1.5, markeredgewidth = 0.5,
                          labels = [r'$10s$', r'$4h$'], figsize = (0.6, 0.7), dpi=600, axwidth = 1.5, fontsize = 10):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n_rules_to_analyse = max(10, n_good_rules)

    eb0 = ax.errorbar([0.1, 1.1], [np.mean(dr[:,2]), np.mean(dr[:,-1])], [np.std(dr[:,2]), np.std(dr[:,-1])],
                linestyle='', marker='.', ms=marker_size_circ, color='grey', zorder=10, markeredgecolor='black', markeredgewidth=markeredgewidth, elinewidth=elinewidth_small)
    eb1 = ax.errorbar([-0.1, 0.9],
                    [np.mean(dr[sorted_inds[:n_rules_to_analyse]][:,2]), np.mean(dr[sorted_inds[:n_rules_to_analyse]][:,-1])],
                    [np.std(dr[sorted_inds[:n_rules_to_analyse]][:,2]), np.std(dr[sorted_inds[:n_rules_to_analyse]][:,-1])],
                    linestyle='', marker='.', ms=marker_size_circ, color='#36454F', zorder=10, markeredgecolor='black', markeredgewidth=markeredgewidth, elinewidth=elinewidth_small)
    for line in eb1[2]:
        line.set_path_effects([pe.Stroke(linewidth=1.5*elinewidth_small, foreground='black'),pe.Normal()])
    for line in eb0[2]:
        line.set_path_effects([pe.Stroke(linewidth=1.5*elinewidth_small, foreground='black'),pe.Normal()])
    # add legend
    ax.legend([eb1, eb0], ['Selected rules', 'All rules'], fontsize=fontsize-2, frameon=False, handletextpad=0.5, labelspacing=0.2, handlelength=1)
    # change location of legend
    legend = ax.get_legend()
    legend.set_bbox_to_anchor((0.5, 1.15))

    ax.set_xlim([-0.4,1.3])
    ax.set_xticks([0,1])
    ax.set_yticks([-0.4, 0, 0.4])
    ax.set_yticklabels([-0.4, 0, 0.4])
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel('A.U.', fontsize=fontsize, labelpad=-13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    plt.show()

def plot_rbg_comparison(d, n_good_rules, sorted_inds, labels = [r'$r_{bg}$'], chosen_time_index = -1, figsize = (0.2, 0.7),
                        dpi=600, axwidth=1, linewidth=1, marker_size_circ = 6, markeredgewidth = 0.5, elinewidth_small = 1.5,
                        fontsize = 10):
    
    n_rules_to_analyse = max(10, n_good_rules)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    d[sorted_inds[:n_rules_to_analyse]]['rate']
    eb0 = ax.errorbar([0.1], np.mean(d['rate'][:,chosen_time_index]), np.std(d['rate'][:,chosen_time_index]),
        linestyle='', marker='.', ms=marker_size_circ, color='grey', zorder=10, markeredgecolor='black', markeredgewidth=markeredgewidth, elinewidth=elinewidth_small)
    eb1 = ax.errorbar([-0.1], np.mean(d['rate'][sorted_inds[:n_rules_to_analyse]][:,chosen_time_index]), np.std(d['rate'][sorted_inds[:n_rules_to_analyse]][:,chosen_time_index]),
                  linestyle='', marker='.', ms=marker_size_circ, color='#36454F', zorder=10, markeredgecolor='black', markeredgewidth=markeredgewidth, elinewidth=elinewidth_small)
    for line in eb1[2]:
        line.set_path_effects([pe.Stroke(linewidth=1.5*elinewidth_small, foreground='black'),pe.Normal()])
    for line in eb0[2]:
        line.set_path_effects([pe.Stroke(linewidth=1.5*elinewidth_small, foreground='black'),pe.Normal()])
    ax.set_xlim([-0.2,0.2])
    ax.set_xticks([0])
    ax.set_yticks([0, 5])
    ax.set_yticklabels([0, 5])
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel('Hz', fontsize=fontsize, labelpad=-8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    plt.show()


### Slice visualization

def string_to_float_list(s):
  if 'nan' in s:
      return [np.nan] * s.count('nan')
  else:
      numbers = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|\d+', s)
      return [float(num) for num in numbers]
  
def plot_rotated_points(rotated_x, rotated_y, dr, inds_stable, xlabel=None, ylabel=None, dpi=600, inds_rules_base=None,
                        figsize=(1,1), xhandlepad=0, font='Arial', fontsize=10, yhandlepad=0, linewidth=1.5, xlim=None, ylim=None, xticks=None, yticks=None, 
                        xticklabels=None, yticklabels=None, vmin=-1, vmax=1,
                        marker_unstable='o', edgecolors_unstable = 'none', marker_size_unstable=0.1, facecolor_unstable='black', linewidths_unstable=0.1,
                        marker_stable='o', edgecolors_stable = 'none', marker_size_stable=0.1, facecolor_stable='black', linewidths_stable=0.1,
                        marker_base='o', edgecolors_base = 'none', marker_size_base=0.1, facecolor_base='black', linewidths_base=0.1,
                        heatmap_label=None, cbarhandlepad=0, cbarticks=[], cbarticklabels=[],
                        ):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(rotated_x[np.invert(inds_stable)], rotated_y[np.invert(inds_stable)], marker=marker_unstable, edgecolors=edgecolors_unstable, 
                     facecolors=facecolor_unstable, linewidths=linewidths_unstable, s=marker_size_unstable)
    
    img = ax.scatter(rotated_x[inds_stable], rotated_y[inds_stable], c=dr[inds_stable], vmin=vmin, vmax=vmax, marker=marker_stable, edgecolors=edgecolors_stable, 
                     facecolors=facecolor_stable, linewidths=linewidths_stable, s=marker_size_stable, cmap=cmp_vis)
    
    if inds_rules_base is not None:
        ax.scatter(rotated_x[inds_rules_base], rotated_y[inds_rules_base], c=dr[inds_rules_base], vmin=vmin, vmax=vmax, marker=marker_base, edgecolors=edgecolors_base, 
                     facecolors=facecolor_base, linewidths=linewidths_base, s=marker_size_base, cmap=cmp_vis)

    if xhandlepad != None:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize, labelpad=xhandlepad)
    else:
        ax.set_xlabel(xlabel, fontname=font, fontsize=fontsize)
        
    if yhandlepad != None:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize, labelpad=yhandlepad)
    else:
        ax.set_ylabel(ylabel, fontname=font, fontsize=fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(width=linewidth, labelsize=fontsize, length=2*linewidth, pad=2)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if xticks != None:
        ax.set_xticks(xticks)
    if xticklabels != None:
        ax.set_xticklabels(xticklabels)
    if yticks != None:
        ax.set_yticks(yticks)
    if yticklabels != None:
        ax.set_yticklabels(yticklabels)

    cbar = fig.colorbar(img, label = heatmap_label, aspect=15, ax=ax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=fontsize, width=linewidth, length=2*linewidth)
    if cbarhandlepad != None:
        cbar.set_label(label=heatmap_label, size=fontsize, labelpad=cbarhandlepad)
    else:
        cbar.set_label(label=heatmap_label, size=fontsize)
    if cbarticks != None:
        cbar.set_ticks(cbarticks)
    if cbarticklabels is not None:
        cbar.set_ticklabels(cbarticklabels)

    plt.show()


### Pong

class ReadoutModel(nn.Module):
    def __init__(self, num_neurons: int, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(num_neurons, out_dim)

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        return self.fc1(rates)

def _load_pong_csvs(ball_csv: str, fr_csv: str, game_index_csv: str):
    df_ball = pd.read_csv(ball_csv)
    ball_positions = torch.tensor(df_ball.values[:, 1:3], dtype=torch.float32)

    df_fr = pd.read_csv(fr_csv)
    firing_rates = torch.tensor(df_fr.values[:, :], dtype=torch.float32).T  # (T, N)

    df_idx = pd.read_csv(game_index_csv)
    start = df_idx.values[:, 1:3]
    end = np.append(df_idx.values[1:, 1:3], (df_idx.values[-1, 1] + 1))
    return ball_positions, firing_rates, start, end

def _normalize_ball_positions(ball_positions: torch.Tensor, grid_size: float = 64.0) -> torch.Tensor:
    return 2 * (ball_positions / grid_size) - 1

def fit_line_and_get_intercept(prediction: torch.Tensor):
    intercepts, slopes = [], []
    for i in range(prediction.shape[0]):
        pred_curr = prediction[i, :2]
        pred_t_minus_3 = prediction[i, 2:4]

        points = torch.stack([pred_curr, pred_t_minus_3])
        x = points[:, 1]
        y = points[:, 0]

        A = torch.stack([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(A, y).solution
        m, c = solution[0], solution[1]

        slopes.append(float(m.item()))
        intercepts.append(float(c.item()))
    return intercepts, slopes

def check_intercept_within_zone(
    prediction: torch.Tensor,
    last_position: torch.Tensor,
    *,
    tolerance: float = 4 / 32,
) -> int:

    intercepts, slopes = fit_line_and_get_intercept(prediction)

    intercept = intercepts[0]
    slope = slopes[0]

    intercept_at_x_minus_1 = slope * (-1) + intercept
    intercept_at_x_minus_1 = max(-1.0, min(intercept_at_x_minus_1, 1.0))

    last_y = float(last_position[0, 0].item())

    intercept_lower = max(-1.0, intercept_at_x_minus_1 - tolerance)
    intercept_upper = min(1.0, intercept_at_x_minus_1 + tolerance)

    last_y_lower = max(-1.0, last_y - tolerance)
    last_y_upper = min(1.0, last_y + tolerance)

    return 1 if (intercept_lower <= last_y_upper and intercept_upper >= last_y_lower) else 0

def train_decode(
    train_ball_csv: str,
    train_fr_csv: str,
    train_game_index_csv: str,
    *,
    lr: float = 1e-3,
    seed: int | None = None,
    device: str | torch.device | None = None,
    save_model: bool = False,
    save_path: str = "model_readout.pt",
    print_every: int = 500,
) -> tuple[nn.Module, dict]:

    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f"Seed:{seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ball_positions, firing_rates, start, end = _load_pong_csvs(
        train_ball_csv, train_fr_csv, train_game_index_csv
    )

    normalized_ball_positions = _normalize_ball_positions(ball_positions)

    firing_rates = firing_rates.to(device)
    normalized_ball_positions = normalized_ball_positions.to(device)

    timesteps, num_neurons = firing_rates.shape
    print(f"Loaded firing rates: {timesteps} timesteps, {num_neurons} neurons")

    model = ReadoutModel(num_neurons=num_neurons, out_dim=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epochs = len(start) - 1

    history = {
        "seed": seed,
        "device": str(device),
        "lr": lr,
        "losses_total": [],
        "losses_current": [],
        "losses_t_minus_3": [],
        "total_steps": 0,
        "epoch_score_running_mean": [],
    }

    accum_total = 0.0
    accum_curr = 0.0
    accum_p3 = 0.0
    total_steps = 0
    total_score = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loss_curr = 0.0
        loss_prev3 = 0.0
        score = 0

        for t in range(int(start[epoch].item()) + 3, int(end[epoch].item())):
            total_steps += 1

            trial_firing_rate = firing_rates[t, :].unsqueeze(0)

            actual_curr = normalized_ball_positions[t, :].unsqueeze(0)
            actual_t_minus_3 = normalized_ball_positions[t - 3, :].unsqueeze(0)

            optimizer.zero_grad()
            prediction = model(trial_firing_rate)

            pred_curr = prediction[:, :2]
            pred_t_minus_3 = prediction[:, 2:4]

            loss_c = loss_fn(pred_curr, actual_curr)
            loss_p3 = loss_fn(pred_t_minus_3, actual_t_minus_3)

            step_loss = loss_c + loss_p3
            step_loss.backward()
            optimizer.step()

            total_loss += float(step_loss.item())
            loss_curr += float(loss_c.item())
            loss_prev3 += float(loss_p3.item())

            last_position = normalized_ball_positions[t, :].unsqueeze(0)
            score = check_intercept_within_zone(prediction.detach(), last_position.detach())

        accum_total += total_loss
        accum_curr += loss_curr
        accum_p3 += loss_prev3

        history["losses_total"].append(accum_total / total_steps)
        history["losses_current"].append(accum_curr / total_steps)
        history["losses_t_minus_3"].append(accum_p3 / total_steps)

        total_score += score
        history["epoch_score_running_mean"].append(total_score / (epoch + 1))

        if (epoch % print_every) == 0:
            print(
                f"Epoch {epoch}, "
                f"Accumulative Mean Total Loss: {history['losses_total'][-1]:.4f}, "
                f"Current Loss: {history['losses_current'][-1]:.4f}, "
                f"t-3 Loss: {history['losses_t_minus_3'][-1]:.4f}, "
                f"Score: {history['epoch_score_running_mean'][-1]:.4f}"
            )

    history["total_steps"] = total_steps
    print("Training complete.")

    if save_model:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "num_neurons": num_neurons,
            "out_dim": 4,
            "seed": seed,
            "lr": lr,
            "history": history,
        }
        torch.save(ckpt, save_path)
        print(f"Saved model checkpoint to: {save_path}")

    return model, history

def test_decode(
    model: nn.Module,
    val_ball_csv: str,
    val_fr_csv: str,
    val_game_index_csv: str,
    *,
    device: str | torch.device | None = None,
) -> dict:

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.eval()

    ball_positions_val, firing_rates_val, start_val, end_val = _load_pong_csvs(
        val_ball_csv, val_fr_csv, val_game_index_csv
    )

    normalized_ball_positions_val = _normalize_ball_positions(ball_positions_val)

    firing_rates_val = firing_rates_val.to(device)
    normalized_ball_positions_val = normalized_ball_positions_val.to(device)

    val_epochs = len(start_val) - 1

    mse_all, rmse_all = [], []
    mse_curr, rmse_curr = [], []
    mse_t_minus_3, rmse_t_minus_3 = [], []

    total_score = 0

    def _compute_metrics(actual: torch.Tensor, pred: torch.Tensor, mse_list: list):
        a = actual.squeeze().detach().cpu().numpy()
        p = pred.squeeze().detach().cpu().numpy()
        mse = mean_squared_error(a, p)
        mse_list.append(float(mse))

    with torch.no_grad():
        for epoch in range(val_epochs):
            score = 0
            for t in range(int(start_val[epoch].item()) + 3, int(end_val[epoch].item())):
                trial_firing_rate = firing_rates_val[t, :].unsqueeze(0)

                actual_curr = normalized_ball_positions_val[t, :].unsqueeze(0)
                actual_t_minus_3 = normalized_ball_positions_val[t - 3, :].unsqueeze(0)

                prediction = model(trial_firing_rate)

                pred_curr = prediction[:, :2]
                pred_t_minus_3 = prediction[:, 2:4]

                _compute_metrics(actual_curr, pred_curr, mse_curr)
                _compute_metrics(actual_t_minus_3, pred_t_minus_3, mse_t_minus_3)

                _compute_metrics(
                    torch.cat([actual_curr, actual_t_minus_3], dim=0),
                    prediction.view(-1, 2),
                    mse_all
                )

                last_position = normalized_ball_positions_val[t, :].unsqueeze(0)
                score = check_intercept_within_zone(prediction, last_position)

            total_score += score

    accuracy = total_score / val_epochs

    def _summary(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"mean": np.nan, "std": np.nan, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }

    results = {
        "accuracy": float(accuracy),

        "mse_all": mse_all,
        "mse_curr": mse_curr,
        "mse_t_minus_3": mse_t_minus_3,

        "summary": {
            "MSE (All Positions)": _summary(mse_all),
            "MSE (Current)": _summary(mse_curr),
            "MSE (T-3)": _summary(mse_t_minus_3),
        }
    }

    print(f"Validation Score: {results['accuracy']:.4f}")
    print("-------------------------------------------------------------")
    for k, v in results["summary"].items():
        print(f"{k:24s} -> Mean: {v['mean']:.2f}, SD: {v['std']:.2f}, n={v['n']}")

    return results

def train_decode_full(
    train_ball_csv: str,
    train_fr_csv: str,
    train_game_index_csv: str,
    *,
    lr: float = 1e-3,
    seed: int | None = None,
    device: str | torch.device | None = None,
    save_model: bool = False,
    save_path: str = "model_readout_full.pt",
    print_every: int = 500,
) -> tuple[nn.Module, dict]:

    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f"Seed:{seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ball_positions, firing_rates, start, end = _load_pong_csvs(
        train_ball_csv, train_fr_csv, train_game_index_csv
    )
    normalized_ball_positions = _normalize_ball_positions(ball_positions)

    firing_rates = firing_rates.to(device)
    normalized_ball_positions = normalized_ball_positions.to(device)

    timesteps, num_neurons = firing_rates.shape
    print(f"Loaded firing rates: {timesteps} timesteps, {num_neurons} neurons")

    out_dim = 10
    model = ReadoutModel(num_neurons=num_neurons, out_dim=out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epochs = len(start) - 1

    history = {
        "seed": seed,
        "device": str(device),
        "lr": lr,
        "losses_total": [],
        "losses_t": [],
        "losses_t_minus_3": [],
        "losses_t_minus_6": [],
        "losses_t_minus_9": [],
        "losses_t_minus_12": [],
        "total_steps": 0,
        "epoch_score_running_mean": [],
    }

    accum_total = 0.0
    accum_t = 0.0
    accum_p3 = 0.0
    accum_p6 = 0.0
    accum_p9 = 0.0
    accum_p12 = 0.0

    total_steps = 0
    total_score = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loss_t = 0.0
        loss_p3 = 0.0
        loss_p6 = 0.0
        loss_p9 = 0.0
        loss_p12 = 0.0
        score = 0

        # Need t-12 available
        for t in range(int(start[epoch].item()) + 12, int(end[epoch].item())):
            total_steps += 1

            trial_firing_rate = firing_rates[t, :].unsqueeze(0)

            actual_t = normalized_ball_positions[t, :].unsqueeze(0)
            actual_p3 = normalized_ball_positions[t - 3, :].unsqueeze(0)
            actual_p6 = normalized_ball_positions[t - 6, :].unsqueeze(0)
            actual_p9 = normalized_ball_positions[t - 9, :].unsqueeze(0)
            actual_p12 = normalized_ball_positions[t - 12, :].unsqueeze(0)

            optimizer.zero_grad()
            prediction = model(trial_firing_rate)

            pred_t = prediction[:, 0:2]
            pred_p3 = prediction[:, 2:4]
            pred_p6 = prediction[:, 4:6]
            pred_p9 = prediction[:, 6:8]
            pred_p12 = prediction[:, 8:10]

            l_t = loss_fn(pred_t, actual_t)
            l_p3 = loss_fn(pred_p3, actual_p3)
            l_p6 = loss_fn(pred_p6, actual_p6)
            l_p9 = loss_fn(pred_p9, actual_p9)
            l_p12 = loss_fn(pred_p12, actual_p12)

            step_loss = l_t + l_p3 + l_p6 + l_p9 + l_p12
            step_loss.backward()
            optimizer.step()

            total_loss += float(step_loss.item())
            loss_t += float(l_t.item())
            loss_p3 += float(l_p3.item())
            loss_p6 += float(l_p6.item())
            loss_p9 += float(l_p9.item())
            loss_p12 += float(l_p12.item())

            last_position = normalized_ball_positions[t, :].unsqueeze(0)
            score = check_intercept_within_zone(prediction.detach(), last_position.detach())

        accum_total += total_loss
        accum_t += loss_t
        accum_p3 += loss_p3
        accum_p6 += loss_p6
        accum_p9 += loss_p9
        accum_p12 += loss_p12

        history["losses_total"].append(accum_total / total_steps)
        history["losses_t"].append(accum_t / total_steps)
        history["losses_t_minus_3"].append(accum_p3 / total_steps)
        history["losses_t_minus_6"].append(accum_p6 / total_steps)
        history["losses_t_minus_9"].append(accum_p9 / total_steps)
        history["losses_t_minus_12"].append(accum_p12 / total_steps)

        total_score += score
        history["epoch_score_running_mean"].append(total_score / (epoch + 1))

        if (epoch % print_every) == 0:
            print(
                f"Epoch {epoch}, "
                f"Acc Mean Total Loss: {history['losses_total'][-1]:.4f}, "
                f"t: {history['losses_t'][-1]:.4f}, "
                f"t-3: {history['losses_t_minus_3'][-1]:.4f}, "
                f"t-6: {history['losses_t_minus_6'][-1]:.4f}, "
                f"t-9: {history['losses_t_minus_9'][-1]:.4f}, "
                f"t-12: {history['losses_t_minus_12'][-1]:.4f}, "
                f"Score: {history['epoch_score_running_mean'][-1]:.4f}"
            )

    history["total_steps"] = total_steps
    print("Training complete.")

    if save_model:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "num_neurons": num_neurons,
            "out_dim": out_dim,
            "seed": seed,
            "lr": lr,
            "history": history,
        }
        torch.save(ckpt, save_path)
        print(f"Saved model checkpoint to: {save_path}")

    return model, history

def test_decode_full(
    model: nn.Module,
    val_ball_csv: str,
    val_fr_csv: str,
    val_game_index_csv: str,
    *,
    device: str | torch.device | None = None,
) -> dict:

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.eval()

    ball_positions_val, firing_rates_val, start_val, end_val = _load_pong_csvs(
        val_ball_csv, val_fr_csv, val_game_index_csv
    )
    normalized_ball_positions_val = _normalize_ball_positions(ball_positions_val)

    firing_rates_val = firing_rates_val.to(device)
    normalized_ball_positions_val = normalized_ball_positions_val.to(device)

    val_epochs = len(start_val) - 1

    mse_all = []
    mse_t, mse_p3, mse_p6, mse_p9, mse_p12 = [], [], [], [], []

    total_score = 0

    def _compute_mse(actual: torch.Tensor, pred: torch.Tensor, mse_list: list):
        a = actual.squeeze().detach().cpu().numpy()
        p = pred.squeeze().detach().cpu().numpy()
        mse_list.append(float(mean_squared_error(a, p)))

    with torch.no_grad():
        for epoch in range(val_epochs):
            score = 0
            for t in range(int(start_val[epoch].item()) + 12, int(end_val[epoch].item())):
                trial_firing_rate = firing_rates_val[t, :].unsqueeze(0)

                actual_t = normalized_ball_positions_val[t, :].unsqueeze(0)
                actual_p3 = normalized_ball_positions_val[t - 3, :].unsqueeze(0)
                actual_p6 = normalized_ball_positions_val[t - 6, :].unsqueeze(0)
                actual_p9 = normalized_ball_positions_val[t - 9, :].unsqueeze(0)
                actual_p12 = normalized_ball_positions_val[t - 12, :].unsqueeze(0)

                prediction = model(trial_firing_rate)

                pred_t = prediction[:, 0:2]
                pred_p3 = prediction[:, 2:4]
                pred_p6 = prediction[:, 4:6]
                pred_p9 = prediction[:, 6:8]
                pred_p12 = prediction[:, 8:10]

                _compute_mse(actual_t, pred_t, mse_t)
                _compute_mse(actual_p3, pred_p3, mse_p3)
                _compute_mse(actual_p6, pred_p6, mse_p6)
                _compute_mse(actual_p9, pred_p9, mse_p9)
                _compute_mse(actual_p12, pred_p12, mse_p12)

                _compute_mse(
                    torch.cat([actual_t, actual_p3, actual_p6, actual_p9, actual_p12], dim=0),
                    prediction.view(-1, 2),
                    mse_all
                )

                last_position = normalized_ball_positions_val[t, :].unsqueeze(0)
                score = check_intercept_within_zone(prediction, last_position)

            total_score += score

    accuracy = total_score / val_epochs

    def _summary(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"mean": np.nan, "std": np.nan, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size),
        }

    results = {
        "accuracy": float(accuracy),
        "mse_all": mse_all,
        "mse_t": mse_t,
        "mse_t_minus_3": mse_p3,
        "mse_t_minus_6": mse_p6,
        "mse_t_minus_9": mse_p9,
        "mse_t_minus_12": mse_p12,
        "summary": {
            "MSE (All Points)": _summary(mse_all),
            "MSE (t)": _summary(mse_t),
            "MSE (t-3)": _summary(mse_p3),
            "MSE (t-6)": _summary(mse_p6),
            "MSE (t-9)": _summary(mse_p9),
            "MSE (t-12)": _summary(mse_p12),
        }
    }

    print(f"Validation Score: {results['accuracy']:.4f}")
    print("-------------------------------------------------------------")
    for k, v in results["summary"].items():
        print(f"{k:18s} -> Mean: {v['mean']:.2f}, SD: {v['std']:.2f}, n={v['n']}")

    return results

def plot_mse_with_std(
    time_points,
    mse_means,
    mse_stds,
    *,
    labels=None,
    colors=None,
    figsize=(15, 3),
    ncols=4,
    ylim=(0, 0.5),
    xlabel="Position (frames)",
    ylabel="MSE test",
    marker_line="-o",
    alpha_fill=0.2,
    hide_spines=True,
):

    K = len(mse_means)
    if labels is None:
        labels = [f"cond_{i}" for i in range(K)]
    if colors is None:
        colors = [None] * K

    plt.figure(figsize=figsize)

    for i in range(K):
        plt.subplot(1, ncols, i + 1)

        y = np.asarray(mse_means[i], dtype=float)
        s = np.asarray(mse_stds[i], dtype=float)

        plt.plot(time_points, y, marker_line, label=labels[i], color=colors[i])
        plt.fill_between(
            time_points,
            y - s,
            y + s,
            color=colors[i],
            alpha=alpha_fill,
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(*ylim)
        plt.title(labels[i])

        if hide_spines:
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_placefield_map(
    csv_path: str,
    *,
    timebin=None,
    timebin_index: int | None = 2,
    grid_size: int = 64,
    normalize_global: bool = True,
    cmap: str = "hot",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize=(5, 4),
    return_grid: bool = False,
):

    data = pd.read_csv(csv_path)

    data['NeuronIndex'] = data['NeuronIndex'].astype(int)
    data = data.sort_values(by='TimeBin').reset_index(drop=True)

    unique_bins = data['TimeBin'].unique()

    if timebin is None:
        if timebin_index is None:
            raise ValueError("Provide either timebin or timebin_index")
        timebin = unique_bins[timebin_index]

    subset = data[data['TimeBin'] == timebin].copy()

    if normalize_global:
        max_rate = data['FiringRate (Hz)'].max()
    else:
        max_rate = subset['FiringRate (Hz)'].max()

    subset['NormalizedFiringRate'] = subset['FiringRate (Hz)'] / max_rate

    grid = np.zeros((grid_size, grid_size))

    for _, row in subset.iterrows():
        neuron_index = int(row['NeuronIndex'])
        if 0 <= neuron_index < grid_size * grid_size:
            x, y = divmod(neuron_index, grid_size)
            grid[x, y] = row['NormalizedFiringRate']

    plt.figure(figsize=figsize)
    im = plt.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized Firing Rate', rotation=270, labelpad=20)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"TimeBin: {timebin}")
    plt.show()

    if return_grid:
        return grid

def get_coordinates(index: int, grid_size: int = 64):
    x = index % grid_size
    y = index // grid_size
    return x, y

def get_sum_weights(weights_file: str, connectivity_file: str, *, grid_size: int = 64):

    weights = pd.read_csv(weights_file)
    connections = pd.read_csv(connectivity_file)

    source_coords = np.array([get_coordinates(i, grid_size) for i in connections["Source"]])
    target_coords = np.array([get_coordinates(i, grid_size) for i in connections["Target"]])

    all_data = []
    for x in range(grid_size):
        source_mask = source_coords[:, 0] == x
        distances = target_coords[source_mask][:, 0] - x

        # assumes weights rows align with connections rows
        weights_at_x = weights[source_mask].values.flatten()

        all_data.append(
            pd.DataFrame(
                {"Source_X": x, "Distance": distances, "Weight": weights_at_x}
            )
        )

    all_data_df = pd.concat(all_data, ignore_index=True)
    sum_weights_df = all_data_df.groupby("Distance")["Weight"].sum().reset_index()
    return all_data_df, sum_weights_df

def plot_distance_weight_profiles(
    file_data,
    *,
    labels=("EE", "EI", "IE", "II"),
    colors=None,
    grid_size: int = 64,
    figsize=(4, 1.5),
    dpi: int = 300,
    axwidth: float = 3,
    fontsize: int = 10,
    xlim=(0, 63),
    xticks=(0, 63),
    ylim=(0, 0.02),
    yticks=(0, 0.02),
    w_pad: float = 0.3,
    normalize: bool = True,
):
    
    K = len(file_data)
    if colors is None:
        colors = [None] * K

    pe1 = [
        pe.Stroke(linewidth=1.5 * 0.66 * axwidth, foreground="black"),
        pe.Stroke(foreground="white", alpha=1),
        pe.Normal(),
    ]

    fig, axes = plt.subplots(1, K, figsize=figsize, dpi=dpi, sharey=True)
    if K == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        _, sum_weights_df = get_sum_weights(
            file_data[i][0], file_data[i][1], grid_size=grid_size
        )

        dists = np.asarray(sum_weights_df["Distance"], dtype=int)
        sum_ws = np.asarray(sum_weights_df["Weight"], dtype=float)

        if normalize:
            denom = np.sum(sum_ws)
            if denom != 0:
                sum_ws = sum_ws / denom

        mid = len(dists) // 2

        pos_distances = dists[mid:]
        neg_dist_w = np.flip(sum_ws[: mid + 1])
        pos_dist_w = sum_ws[mid:]

        ax.plot(
            pos_distances,
            pos_dist_w,
            linewidth=axwidth,
            color=colors[i],
            clip_on=True,
            zorder=0.5,
        )

        ax.plot(
            pos_distances,
            neg_dist_w,
            "-",
            linewidth=0.66 * axwidth,
            color=colors[i],
            clip_on=True,
            zorder=0.6,
            path_effects=pe1,
            alpha=0.4,
        )

        ax.set_xlim(list(xlim))
        ax.set_xticks(list(xticks))
        ax.set_title(labels[i], fontsize=fontsize, pad=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(axwidth)
        ax.spines["left"].set_linewidth(axwidth)

        ax.tick_params(width=axwidth, labelsize=fontsize, length=2 * axwidth, pad=1)
        ax.set_xlabel("d", fontsize=fontsize)

    axes[0].set_ylabel(r"$\langle w_{ij}(i \rightarrow j = d) \rangle$", fontsize=fontsize)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    axes[0].set_ylim(list(ylim))
    axes[0].set_yticks(list(yticks))

    plt.tight_layout(w_pad=w_pad)
    plt.show()

def plot_accuracy_bar(
    file_paths,
    *,
    column="val_accuracy",
    labels=None,
    colors=None,
    figsize=(4, 4),
    ylabel="Pong Accuracy",
    ylim=(0, 0.8),
    capsize=5,
    show_values=False,
):

    dfs = [pd.read_csv(fp) for fp in file_paths]

    means = [df[column].mean() for df in dfs]
    stds = [df[column].std() for df in dfs]

    if labels is None:
        labels = [f"cond_{i}" for i in range(len(file_paths))]
    if colors is None:
        colors = [None] * len(file_paths)

    plt.figure(figsize=figsize)
    bars = plt.bar(labels, means, yerr=stds, capsize=capsize, color=colors)

    if show_values:
        for bar, val in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.ylabel(ylabel)
    plt.ylim(*ylim)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


### Demo

def plot_engrams_demo(sts):
    n_recorded = 4096
    l_stim_on_pretraineng = 1
    l_stim_off_pretraineng = 1
    l_pre_test_record = 1
    frac_size_engram = 0.05
    lpt = 30
    lt = 50
    n_fam_stim = 5
    n_nov_stim = 2
    n_tot_stim = 7
    ontime_test = 0.2
    offtime_test = 2.8
    nseqs_test = 15
    seq_length_test = 4
    break_durations = [1, 9, 10, 40, 60, 180, 300, 600,  2400, 10800]
    n_tests = len(break_durations)
    bin_size_big = 0.1
    l_pretraineng_tot = n_tot_stim*(l_stim_on_pretraineng+l_stim_off_pretraineng)
    l_singlestims_test = n_tot_stim*(ontime_test + offtime_test)
    l_seqstims_test = nseqs_test*(seq_length_test*ontime_test+offtime_test)
    l_1test = l_singlestims_test + l_seqstims_test
    test_starts = np.zeros(len(break_durations))
    test_starts[0] = lpt + l_pretraineng_tot + lt + break_durations[0]
    for i in range(1,len(break_durations)):
        test_starts[i] = test_starts[i-1] + break_durations[i] + l_1test
    engram_neurons = get_engram_neurons_npy(n_tot_stim, lpt, l_stim_on_pretraineng, l_stim_off_pretraineng, frac_size_engram, n_recorded, sts)
    eng_rates = get_eng_rate_npy(sts, test_starts, l_pre_test_record, l_1test, bin_size_big, n_tests, n_recorded, n_tot_stim, engram_neurons)

    #make the ordering per engram. problem, some neurons can be assigned to 2 engrams, thouhg it is rare.
    # assign a unique label to each neuron
    neuron_label = np.zeros(n_recorded) + 7 #0->nov1, ... 6->fam5, 7 unassigned
    for i in range(n_recorded):
        found = False
        eng_count = 0
        while (not found) and (eng_count <= 6):
            if len(np.argwhere(engram_neurons[eng_count, :] == i))>0:
                found = True
                neuron_label[i] = eng_count
            eng_count += 1

    test_sessions = [1, 9]
    neuron_label_bis = np.array(neuron_label) # simply rearranging label names for plotting purposes
    neuron_label_bis[np.where(neuron_label_bis==0)[0]] = -1
    neuron_label_bis[np.where(neuron_label_bis==1)[0]] = -1
    neuron_label_bis[np.where(neuron_label_bis==2)[0][100:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==2)[0][:100]] = 0

    neuron_label_bis[np.where(neuron_label_bis==3)[0][100:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==3)[0][:100]] = 1

    neuron_label_bis[np.where(neuron_label_bis==4)[0][100:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==4)[0][:100]] = 2

    neuron_label_bis[np.where(neuron_label_bis==5)[0][100:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==5)[0][:100]] = 3

    neuron_label_bis[np.where(neuron_label_bis==6)[0][100:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==6)[0][:100]] = 4

    neuron_label_bis[np.where(neuron_label_bis==7)[0][200:]] = -1
    neuron_label_bis[np.where(neuron_label_bis==7)[0][:200]] = 5
    test_session = 2
    print('After 10s')
    plot_raster_w_engrams_sep_background(sts=sts,
        neuron_labels=neuron_label_bis,
        n_recorded=n_recorded,
        colors_label = ['#36454F', color_nov1, color_fam1, color_fam2, color_fam3, color_fam4, color_fam5, '#36454F'],
        colors_raster = [color_fam1, color_fam2, color_fam3, color_fam4, color_fam5, '#36454F'],
        x_lim = [test_starts[test_session]+2.5, test_starts[test_session]+8.9],
        markersize=0.1,
        lag_engr_bg = 30,
        t_start_each_stim = [i*(ontime_test + offtime_test) for i in range(7)] + test_starts[test_session],
        ontime = ontime_test,
        linewidth_stim_line=1.5,
        y_stim_line = 730,
        figsize= (0.8,0.7), #(2.1,0.7), (0.8,0.7) (1.5,0.7)
        x_label= "200ms", #r'$4$' + "h"
        x_ticks= [],
        x_ticklabels= [],
        y_ticks=[],
        y_label="100 neurons",
        y_lim=[0,800], #1400
        fontsize=10,
        dpi=600,
        ylabel_xloc=0.0,
        ylabel_yloc=0.0,
        xlabel_xloc=0.40,
        xlabel_yloc=0.92,
        y_bar_xloc=-0.05,
        y_bar_ylocs=[2.85/7, 3.6/7],
        axwidth=1.5);plt.show()
    
    test_session = 9
    print('After 4h')
    plot_raster_w_engrams_sep_background(sts=sts,
        neuron_labels=neuron_label_bis,
        n_recorded=n_recorded,
        colors_label = ['#36454F', color_nov1, color_fam1, color_fam2, color_fam3, color_fam4, color_fam5, '#36454F'],
        colors_raster = [color_fam1, color_fam2, color_fam3, color_fam4, color_fam5, '#36454F'],
        x_lim = [test_starts[test_session]+2.5, test_starts[test_session]+8.9],
        markersize=0.1,
        lag_engr_bg = 30,
        t_start_each_stim = [i*(ontime_test + offtime_test) for i in range(7)] + test_starts[test_session],
        ontime = ontime_test,
        linewidth_stim_line=1.5,
        y_stim_line = 730,
        figsize= (0.8,0.7), #(2.1,0.7), (0.8,0.7) (1.5,0.7)
        x_label= "200ms", #r'$4$' + "h"
        x_ticks= [],
        x_ticklabels= [],
        y_ticks=[],
        y_label="100 neurons",
        y_lim=[0,800], #1400
        fontsize=10,
        dpi=600,
        ylabel_xloc=0.0,
        ylabel_yloc=0.0,
        xlabel_xloc=0.40,
        xlabel_yloc=0.92,
        y_bar_xloc=-0.05,
        y_bar_ylocs=[2.85/7, 3.6/7],
        axwidth=1.5);plt.show()
    
def plot_dr_demo(dr, color="black", x_label='time', y_label=r'$\Delta r_{mem}$', x_lim = None, y_lim=None,
                linewidth=1.5, axwidth=1.5, fontsize=10, figsize=(3, 1), font = "arial",
                dpi=600, x_ticks=[0,5,9], x_ticklabels=["1s","5min","4h"], y_ticks=None, y_ticklabels=None):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(dr, color=color, linewidth=linewidth, marker='')

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label, fontsize=fontsize, fontname=font, labelpad = 0)
    if x_lim is not None:
        ax.set_xlim([t_lim[0], t_lim[1]])
    
    if y_lim is not None:
        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_yticks([y_lim[0], y_lim[1]])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel(y_label, fontname=font, fontsize=fontsize, labelpad = 0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(axwidth)
    ax.spines['left'].set_linewidth(axwidth)
    ax.tick_params(width=axwidth, labelsize=fontsize, length=2*axwidth)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(font)
    plt.show()