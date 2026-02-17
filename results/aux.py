import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


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


### Spiking network analysis, compute engrams

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

def get_engram_neurons(n_tot_stim, lpt, l_stim_on_pretraineng, l_stim_off_pretraineng, frac_size_engram, n_recorded, spiketimes):
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
        rs = get_individual_rates(start_times_stim_pretraineng[stim_num], 
                                start_times_stim_pretraineng[stim_num] + l_stim_on_pretraineng,
                                spiketimes, n_recorded)
        sorted_array = np.argsort(rs)
        engrams[stim_num] = sorted_array[-n_neurons_per_engrams:]
    return(engrams)

def get_individual_rates(start, stop, spiketimes, n_recorded):
    """
    Compute the the firing of each recorded (excitatory) neuron 
    between start and stop
    """
    rates = np.zeros(n_recorded)
    for neuron in range(n_recorded):
        rates[neuron] =  np.sum(np.logical_and(start<=spiketimes[str(neuron)], 
                                            spiketimes[str(neuron)]<=stop))/(stop-start)
    return(rates)