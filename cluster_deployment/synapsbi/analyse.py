import numpy as np
import h5py
from itertools import chain
from scipy.special import rel_entr
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
import sys
import torch
import spikeye.analyze as a
import time
import matplotlib.pyplot as plt

def default_x():
        """
        Make dictionary to sample values per metric to get filtered posterior samples.

        The key for each entry corresponds to a metric defined in ComputeMetrics.
        The item of each entry is a function that takes an integer `num` as input
            and returns `num` values of the metric.
        We can use these values as conditional values corresponding to which we
            sample parameters from the density estimator.
        """
        # TODO need entry here for every metric we intend to use for filtering
        # ["rate","cv_isi","kl_isi","spatial_Fano","temporal_Fano","auto_cov","fft",
#                    "w_blow", "std_rate_temporal","std_rate_spatial","std_cv"]


        return {"rate": lambda num: torch.rand(num, 1) * 29 + 1, #]1,50]
                "cv_isi": lambda num: torch.rand(num, 1) * 2 + .7, #[0.7,2.7]
                "kl_isi": lambda num: torch.rand(num, 1) * .5, #[0,0.5]
                "spatial_Fano": lambda num: torch.rand(num, 1) * 2 + 0.5, #[0.5,2.5]
                "temporal_Fano": lambda num: torch.rand(num, 1) * 2 + 0.5, #[0.5,2.5]
                "auto_cov": lambda num: torch.rand(num, 1) * .1, #[0,0.1]
                "fft": lambda num: torch.rand(num, 1), #[0,1]
                "w_blow": lambda num: torch.rand(num, 1) * .1, #[0,0.1]
                "std_rate_temporal": lambda num: torch.rand(num, 1) * .05, #[0,0.05]
                "std_rate_spatial": lambda num: torch.rand(num, 1) * 5, #[0,5]
                "std_cv": lambda num: torch.rand(num, 1)*0.2, #[0,0.2]
                "w_creep": lambda num: torch.rand(num, 1)*0.05, #[0,0.05]
                "rate_i": lambda num: torch.rand(num, 1)*49 + 1, #[1,50]
                "weef": lambda num: torch.rand(num, 1)*0.5, #[0,0.5]
                "weif": lambda num: torch.rand(num, 1)*0.5, #[0,0.5]
                "wief": lambda num: torch.rand(num, 1)*5, #[0,5]
                "wiif": lambda num: torch.rand(num, 1)*5, #[0,5]
                "r_fam": lambda num: torch.rand(num, 1)*1 + 13, #[13,14]
                "r_nov": lambda num: torch.rand(num, 1)*1 + 15, #[15, 16]
                "std_fam": lambda num: torch.rand(num, 1)*2 + 15, #[15,17]
                "std_nov": lambda num: torch.rand(num, 1)*2 + 19} #[19, 21]

def condition():
    """
    Make dictionary of metric-specific conditions to rejection sample simulations.

    The key for each entry corresponds to a metric defined in ComputeMetrics.
    The item of each entry is a function that takes a metric value as input
        and returns boolean indicating if it satisfies a specified condition.
    We can use these to rejection sample and reuse simulations from previous rounds.
    """
    # TODO need entry here for every metric we intend to use for filtering
    return {"rate": lambda x: np.logical_and(1 <= x, x <= 50),
            "cv_isi": lambda x: np.logical_and(0.7 <= x, x <= 2.7),
            "kl_isi": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "spatial_Fano": lambda x: np.logical_and(0.5 <= x, x <= 2.5),
            "temporal_Fano": lambda x: np.logical_and(0.5 <= x, x <= 2.5),
            "auto_cov": lambda x: np.logical_and(0 <= x, x <= 0.1),
            "fft": lambda x: np.logical_and(0 <= x, x <= 1),
            "w_blow": lambda x: np.logical_and(0 <= x, x <= 0.1),
            "std_rate_temporal": lambda x: np.logical_and(0 <= x, x <= 0.05),
            "std_rate_spatial": lambda x: np.logical_and(0 <= x, x <= 5),
            "std_cv": lambda x: np.logical_and(0 <= x, x <= 0.2),
            "w_creep": lambda x: np.logical_and(0 <= x, x <= 0.05),
            "rate_i": lambda x: np.logical_and(1 <= x, x <= 50),
            "weef": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "weif": lambda x: np.logical_and(0 <= x, x <= 0.5),
            "wief": lambda x: np.logical_and(0 <= x, x <= 5),
            "wiif": lambda x: np.logical_and(0 <= x, x <= 5),
            "r_fam": lambda x: np.logical_and(13 <= x, x <= 14),
            "r_nov": lambda x: np.logical_and(15 <= x, x <= 16),
            "std_fam": lambda x: np.logical_and(15 <= x, x <= 17)}


def get_w_distr(w_dict=None, t_start=0, t_stop=60):
    # check that t_start and t_stop are legit
    if t_start > w_dict['t'][-1] or t_stop < w_dict['t'][0]:
        raise ValueError('t_start or t_stop are outside the recorded range')
    valid_ts = [t_start<i<t_stop for i in w_dict['t']]
    return(w_dict['w'][:, valid_ts].flatten())



class ComputeMetrics:
    """Compute metrics given a simulation."""

    def __init__(self, spiketimes: dict, sim_params: dict, weights: dict=None, spiketimes_i: dict=None) -> None:
        """Set up class to compute metrics."""
        # initialise an object directly with spiketimes dict, weight dict and
        # params dict
        self.spiketimes = spiketimes
        self.spiketimes_i = spiketimes_i
        self.params = sim_params
        self.weights = weights

        self.binned_spikes_small_computed = False
        self.binned_spikes_small = np.array([])
        self.ts_small = np.array([])
        self.binned_spikes_medium_computed = False
        self.binned_spikes_medium = np.array([])
        self.binned_spikes_big_computed = False
        self.binned_spikes_big = np.array([])

        self.isis_computed = False
        self.isis = dict()

        self.cvs_computed = False
        self.cvs = np.zeros(self.params["n_recorded"])

    def _check(self, keys):
        assert np.all([k in self.params.keys() for k in keys])

    def _return_nan(metric_func):
        def modify_metric_to_return_nan(self):
            if self.spiketimes is None or self.weights is None:
                return np.nan
            else:
                return metric_func(self)
        return modify_metric_to_return_nan

    def get_binned_spikes_small(self):
        if not self.binned_spikes_small_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_small"])
            self.binned_spikes_small = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.ts_small = bins[:-1]
            self.binned_spikes_small_computed = True
        return(self.binned_spikes_small)
    
    def get_binned_spikes_medium(self):
        if not self.binned_spikes_medium_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_medium"])
            self.binned_spikes_medium = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.binned_spikes_medium_computed = True
        return(self.binned_spikes_medium)
    
    def get_binned_spikes_big(self):
        if not self.binned_spikes_big_computed:
            bins = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_big"])
            self.binned_spikes_big = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                    for neuron_num in range(self.params["n_recorded"])])
            self.binned_spikes_big_computed = True
        return(self.binned_spikes_big)
    
    def get_autocov(self, neuron_num):
        lags = int(self.params["window_view_auto_cov"]/self.params["bin_size_medium"])
        x =  self.binned_spikes_medium[neuron_num, :]
        if np.std(x) > 0.05: #tuned so that a spiketrain of 1Hz poisson with bin 10ms ish is not detected
            xcorr = correlate(x - x.mean(), x - x.mean(), 'full')
            xcorr = np.abs(xcorr[:]) / xcorr.max()
            return(np.mean(xcorr[(xcorr.size//2-(lags)):(xcorr.size//2+(lags+1))]))
        else: #signal is almost constant
            return(1)

    def get_isis(self):
        if not self.isis_computed:
            for neuron_num in range(self.params["n_recorded"]):
                self.isis[str(neuron_num)] = np.diff(self.spiketimes[str(neuron_num)])
            self.isis_computed = True
        return(self.isis)

    def get_cvs(self):
        self.get_isis()
        if not self.cvs_computed:
            for neuron_num in range(self.params["n_recorded"]):
                if len(self.isis[str(neuron_num)]) > 2:
                    self.cvs[neuron_num] = np.std(self.isis[str(neuron_num)])/np.mean(self.isis[str(neuron_num)])
            self.cvs_computed = True
        return(self.cvs)

    def get_isi_aggregate(self):
        return(np.array(list(chain(*self.get_isis().values()))))

    def compute_fano(self, counts):
        if np.sum(counts) <= 3:
            return(0)
        else: 
            return(np.var(counts)/np.mean(counts))

    def get_pop_rate_square_window(self, which="exc"):
        ts = np.arange(self.params["t_start_rec"], self.params["t_stop_rec"], self.params["bin_size_big"])
        rates = np.zeros((self.params["n_recorded"], len(ts)-1)) 
        if which == "exc":
            for neuron in range(self.params["n_recorded"]):
                inds_insert = np.searchsorted(self.spiketimes[str(neuron)], ts, side='left', sorter=None)
                rates[neuron] = np.diff(inds_insert)
        else:
            for neuron in range(self.params["n_recorded_i"]):
                inds_insert = np.searchsorted(self.spiketimes_i[str(neuron)], ts, side='left', sorter=None)
                rates[neuron] = np.diff(inds_insert)
        
        pop_rate = np.zeros(len(ts)-1)
        for i in range(len(ts)-1):
            pop_rate[i] = np.mean(rates[:,i])
        return(pop_rate/self.params["bin_size_big"])

    @property
    @_return_nan
    def rate(self):
        """Total population rate."""
        self._check(["n_recorded", "ls"])
        n_tot_spikes = np.sum([i.shape[0] for i in self.spiketimes.values()])
        return(n_tot_spikes/self.params["n_recorded"]/self.params["ls"])

    @property
    @_return_nan
    def cv_isi(self):
        """Coefficient of variation of exc neurons' interspike-interval distribution, averaged over neurons"""
        self._check(["n_recorded"])
        return(np.mean(self.get_cvs()))

    @property
    @_return_nan
    def kl_isi(self):
        """KL divergence between Poisson and simulated spike ISI distribution (single isi distr aggregated over all neurons)"""
        self._check(
            ["n_recorded",
             "t_start_rec",
             "t_stop_rec",
             "n_bins_kl_isi",
             "isi_lim_kl_isi"])

        isis_agg = self.get_isi_aggregate()
        xs = np.linspace(self.params["isi_lim_kl_isi"][0], self.params["isi_lim_kl_isi"][1],num=self.params["n_bins_kl_isi"])
        binned_isi,_ = np.histogram(np.clip(isis_agg, self.params["isi_lim_kl_isi"][0], self.params["isi_lim_kl_isi"][1]),
                                                bins=xs,
                                                density=False)
        binned_isi = binned_isi/max(len(isis_agg), 1)
        if self.rate <= 0.1:
            rate = 0.01
        else:
            rate = 1/np.mean(isis_agg)
        binned_poisson_isi = np.array([np.exp(-rate*xs[i-1]) - np.exp(-rate*xs[i]) for i in range(1, len(xs))])
        kl = np.sum(rel_entr(binned_isi, binned_poisson_isi))
        return(np.abs(kl))

    @property
    @_return_nan
    def spatial_Fano(self):
        """Fano factor: computed over neurons during a single timebin, then returns an average over timebins"""
        self._check(
            ["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_big"])

        self.get_binned_spikes_big()
        return(np.mean([self.compute_fano(self.binned_spikes_big[:,i]) for i in range(len(self.binned_spikes_big[0]))]))

    @property
    @_return_nan
    def temporal_Fano(self):
        """Fano factor: computed over timebins in single neurons (temporal), then returns an average over neurons"""
        self._check(
            ["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_big"])

        self.get_binned_spikes_big()
        return(np.mean([self.compute_fano(self.binned_spikes_big[i,:]) for i in range(self.params["n_recorded"])]))

    @property
    @_return_nan
    def auto_cov(self):
        """fraction of non zero elements in the autocovariance of spiketrains, averaged over neurons"""
        self._check(
            ["n_recorded",
             "t_start_rec",
             "t_stop_rec",
             "bin_size_medium",
             "window_view_auto_cov"])

        self.get_binned_spikes_medium()
        return(np.mean([self.get_autocov(i) for i in range(self.params["n_recorded"])]))

    @property
    @_return_nan
    def fft(self):
        """area under the curve in fourier transform, computed over pop firing rate with small bin"""
        self._check(["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_small"])

        self.get_binned_spikes_small()
        xf = fftfreq(len(self.ts_small), self.params["bin_size_small"])[:len(self.ts_small)//2]
        yf = 2.0/len(self.ts_small) * np.abs( fft(np.mean(self.binned_spikes_small, axis=0))[0:len(self.ts_small)//2])
        return(np.sum(yf[1:]))

    @property
    @_return_nan
    def w_blow(self):
        """Indicate if synaptic weights have exploded."""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "t_start_rec", "t_stop_rec", "wmax"])

        if self.weights is None:
            return(-1000)

        f_blow = 0
        for key in self.weights.keys():
            if key != "t":
                w_distr = a.get_w_distr(w_dict={"w": self.weights[key], "t": self.weights["t"]}, t_start=self.params["t_start_rec"], t_stop=self.params["t_stop_rec"])
                f_blow += np.sum([i == 0 or i == self.params["wmax"] for i in w_distr]) / len(w_distr)
        return (f_blow / (len(self.weights.keys()) - 1))
    
    @property
    @_return_nan
    def std_rate_temporal(self):
        """compute std of population firing rate across time"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "t_start_rec", "t_stop_rec"])
        self.get_binned_spikes_small()
        return(np.std(np.mean(self.binned_spikes_small, axis=0)))
    
    @property
    @_return_nan
    def std_rate_spatial(self):
        """compute std of individual firing rates"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded", "ls"])
        all_rates = np.array( [len(self.spiketimes[str(j)])/self.params["ls"] for j in range(self.params["n_recorded"])] ) 
        return( np.std(all_rates) )
    
    @property
    @_return_nan
    def std_cv(self):
        """compute std of individual neurons cv_isi"""
        # check that the simulation has the params for the metric to be computed
        self._check(["n_recorded"])
        return( np.std(self.get_cvs()) )
    
    @property
    @_return_nan
    def w_creep(self):
        """compute change of mean weight between start and finish (as percentage), max amount all weights considered"""
        # check that the simulation has the params for the metric to be computed
        self._check(["t_start_rec", "t_stop_rec"])
        w_creep_metric = 0
        for key in self.weights.keys():
            if key != "t":
                start_w = np.mean(self.weights[key][:,0])
                end_w = np.mean(self.weights[key][:,-1])
                if start_w + end_w > 0.1:
                    candidate = np.abs(2*(end_w - start_w)/(end_w + start_w))
                    if candidate > w_creep_metric:
                        w_creep_metric = candidate
        return( w_creep_metric )
    
    @property
    @_return_nan
    def rate_i(self):
        """Total population rate of inh population"""
        self._check(["n_recorded_i", "ls"])
        if self.spiketimes_i is None:
            return(np.nan)
        n_tot_spikes = np.sum([i.shape[0] for i in self.spiketimes_i.values()])
        return(n_tot_spikes/self.params["n_recorded_i"]/self.params["ls"])
    
    @property
    @_return_nan
    def weef(self):
        """final mean EE weight"""
        return(np.mean(self.weights["ee"][:,-1]))
    
    @property
    @_return_nan
    def weif(self):
        """final mean EI weight"""
        return(np.mean(self.weights["ei"][:,-1]))
    
    @property
    @_return_nan
    def wief(self):
        """final mean IE weight"""
        return(np.mean(self.weights["ie"][:,-1]))
    
    @property
    @_return_nan
    def wiif(self):
        """final mean II weight"""
        return(np.mean(self.weights["ii"][:,-1]))
    
    @property
    @_return_nan
    def r_nov(self):
        """pop rate in response to nov stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"]
        stop = start + self.params["lp"]
        return(np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def r_fam(self):
        """pop rate in response to familiar stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"] + self.params["lp"] + self.params["lb2"]
        stop = start + self.params["lp"]
        return(np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def std_nov(self):
        """std of individual neurons rate in response to nov stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"]
        stop = start + self.params["lp"]
        return(np.std([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def std_fam(self):
        """std of individual neurons rate in response to familiar stimulus, only relevant to BND task"""
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        start = self.params["lpt"] + self.params["lt"] + self.params["lb0"] + self.params["lb1"] + self.params["lp"] + self.params["lb2"]
        stop = start + self.params["lp"]
        return(np.std([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()]))
    
    @property
    @_return_nan
    def ratio_nov_fam(self):
        self._check(["n_recorded", "lpt", "lt", "lb0", "lb1", "lp", "lb2"])
        return(self.r_nov - self.r_fam)/(self.r_fam+0.0001)

    @property
    @_return_nan
    def prate_sqw_e(self):
        self._check(["n_recorded", "t_start_rec", "t_stop_rec", "bin_size_big"])
        return(self.get_pop_rate_square_window(which="exc"))
    
    @property
    @_return_nan
    def prate_sqw_i(self):
        self._check(["n_recorded_i", "t_start_rec", "t_stop_rec", "bin_size_big"])
        return(self.get_pop_rate_square_window(which="inh"))

    
    
class ComputeMetrics_seq:
    """Compute metrics given a simulation of the sequential task exclusively"""

    def __init__(self, spiketimes: dict, 
                 sim_params: dict, 
                 hard_coded_sim_params: dict,
                 weights: dict=None, 
                 spiketimes_i: dict=None) -> None:
        """Set up class to compute metrics."""
        # initialise an object directly with spiketimes dict, weight dict and
        # params dict
        self.spiketimes = spiketimes
        self.spiketimes_i = spiketimes_i
        self.params = sim_params
        for key in hard_coded_sim_params.keys():
            self.params[key] = hard_coded_sim_params[key]
        
        # "l_stim_on_pretraineng" 1
        # "l_stim_off_pretraineng" 1 
        # "frac_size_engram" 0.1 
        # "n_fam_stim" 5
        # "n_nov_stim" 2
        # "n_tot_stim" 7
        # "ordering_pretraineng" n1 n2 f1 f2 f3 f4 f5
        # "ordering_test_isolation" n2 n1 f2 f5 f3 f4 f1
        # "break_durations" [1, 9,  10, 40, 60,  180, 300, 600,  2400, 10800]
        # "l_pre_test_record" 1
        # "seq_length_test" 4
        # "nseqs_test" 15
        # "n_recorded", "n_recorded", "lpt", "lt", 
        #"ontime_test", "offtime_test", "bin_size_big"

        self.weights = weights

        self.binned_spikes_small_computed = False
        self.binned_spikes_medium_computed = False
        self.binned_spikes_big_computed = False
        self.binned_pop_rate_computed = False
        
        self.get_test_phase_start_times()
        self.get_engram_neurons()
        self.n_tests = len(self.params["break_durations"])

    def _check(self, keys):
        assert np.all([k in self.params.keys() for k in keys])

    def _return_nan(metric_func):
        def modify_metric_to_return_nan(self):
            if self.spiketimes is None or self.weights is None:
                return np.nan
            else:
                return metric_func(self)
        return modify_metric_to_return_nan

    def get_individual_rates(self, start, stop):
        """
        Compute the the firing of each recorded (excitatory) neuron 
        between start and stop
        """
        self._check(["n_recorded"])
        rates = np.zeros(self.params["n_recorded"])
        for neuron in range(self.params["n_recorded"]):
            rates[neuron] =  np.sum(np.logical_and(start<=self.spiketimes[str(neuron)], 
                                                   self.spiketimes[str(neuron)]<=stop))/(stop-start)
        return(rates)

    def get_engram_neurons(self):
        #goal: get the top frac_size most active neurons during presentation 
        #of each stimulus before we even start the task
        self._check(["n_recorded", "l_stim_on_pretraineng", 
                     "frac_size_engram", "lpt", "n_fam_stim",
                     "n_nov_stim", "n_tot_stim"])

        #get the start times of each stimulus during pretrain_assessment
        start_times_stim_pretraineng = np.zeros(self.params["n_tot_stim"])
        start_times_stim_pretraineng[0] = self.params["lpt"]
        for i in range(1,self.params["n_tot_stim"]):
            start_times_stim_pretraineng[i] = start_times_stim_pretraineng[i-1] + self.params["l_stim_on_pretraineng"] + self.params["l_stim_off_pretraineng"]

        n_engrams = self.params["n_tot_stim"]
        n_neurons_per_engrams = int(self.params["frac_size_engram"]*self.params["n_recorded"])
        self.engrams = np.zeros((n_engrams,n_neurons_per_engrams), dtype=int)
        if self.spiketimes is None:
            return

        #for each stimulus, pick the top N most active neurons (given by frac_size)
        for stim_num in range(self.params["n_tot_stim"]):
            #get individual firing rates
            rs = self.get_individual_rates(start_times_stim_pretraineng[stim_num], 
                                           start_times_stim_pretraineng[stim_num] + self.params["l_stim_on_pretraineng"])
            sorted_array = np.argsort(rs)
            self.engrams[stim_num] = sorted_array[-n_neurons_per_engrams:]
    
    def get_test_phase_start_times(self):
        self.l_pretraineng_tot = self.params["n_tot_stim"]*(self.params["l_stim_on_pretraineng"] 
                               + self.params["l_stim_off_pretraineng"])
        
        self.l_singlestims_test = self.params["n_tot_stim"]*(self.params["ontime_test"] 
                                                        + self.params["offtime_test"])
        self.l_seqstims_test = self.params["nseqs_test"]*(self.params["seq_length_test"]*self.params["ontime_test"] + self.params["offtime_test"])
        self.l_1test = self.l_singlestims_test + self.l_seqstims_test
        self.test_starts = np.zeros(len(self.params["break_durations"]))
        self.test_starts[0] = self.params["lpt"] + self.l_pretraineng_tot + self.params["lt"] + self.params["break_durations"][0]
        for i in range(1,len(self.params["break_durations"])):
            self.test_starts[i] = self.test_starts[i-1] + self.params["break_durations"][i] + self.l_1test

    def get_binned_spikes_big(self):
        """
        computes binned_spikes for each recorded neuron. 
        Since we have several test sessions during a single simulation, we use a type
        ts = np.array[n_test_phases, n_bins]
        binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
        """
        if not self.binned_spikes_big_computed: 
            n_bins_per_test = len(np.arange(self.test_starts[0]-self.params["l_pre_test_record"], self.test_starts[0]+self.l_1test, self.params["bin_size_big"]))-1
            self.ts_big = np.zeros((self.n_tests, n_bins_per_test))
            self.binned_spikes_big = np.zeros((self.n_tests, self.params["n_recorded"], n_bins_per_test))
            for i in range(self.n_tests):
                bins = np.arange(self.test_starts[i]-self.params["l_pre_test_record"], self.test_starts[i]+self.l_1test, self.params["bin_size_big"])
                self.binned_spikes_big[i] = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                        for neuron_num in range(self.params["n_recorded"])])
                self.ts_big[i] = bins[:-1]
            self.binned_spikes_big_computed = True

    def get_binned_pop_rate(self):
        """
        computes the population firing rate for all test sessions with bin_size_big
        ts = np.array[n_test_phases, n_bins]
        binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
        """   
        if not self.binned_pop_rate_computed: 
            self.get_binned_spikes_big()
            n_bins_per_test = len(np.arange(self.test_starts[0]-self.params["l_pre_test_record"], self.test_starts[0]+self.l_1test, self.params["bin_size_big"]))-1
            self.binned_pop_rate = np.zeros((self.n_tests, n_bins_per_test))
            for i in range(self.n_tests):
                bins = np.arange(self.test_starts[i]-self.params["l_pre_test_record"], self.test_starts[i]+self.l_1test, self.params["bin_size_big"])
                self.binned_pop_rate[i] = np.mean(self.binned_spikes_big[i], axis=0)
            self.binned_pop_rate = self.binned_pop_rate/self.params["bin_size_big"]
            self.binned_pop_rate_computed = True
    
    def get_inds_single_stim_pres(self, test_num):
        '''
        using self.ts_big at test sessions test_num,
        gives the indices of ts_big when each stimulus
        is being presented in single succession (using the "ordering_pretrain_eng" ordering)
        returns: numpy array [n_tot_stim, n_bins] 1 if index to select, 0 otherwise. dtype=bool
        will be used for logical indexing after
        '''
        inds_single_stim = np.zeros( (self.params["n_tot_stim"],len(self.ts_big[0])), dtype=bool )
        for i in range(self.params["n_tot_stim"]):
            label = self.params["ordering_pretraineng"][i]
            ind_label_pretrain_eng = np.where([label in i for i in self.params["ordering_test_isolation"]])[0][0]
            start = self.test_starts[test_num] + ind_label_pretrain_eng*(self.params["ontime_test"] + self.params["offtime_test"])
            stop = start + self.params["ontime_test"]
            inds_single_stim[i] = np.logical_and(start <= self.ts_big[test_num], self.ts_big[test_num] <= stop)
        return(inds_single_stim)
    
    def get_responses_engram_each_stim(self, inds_stim, test_num):
        '''
        Inputs:
            inds_stim: [n_engrams, n_timepoints] which timepoints correspond to presentation of each stimulus:  
                inds_stim[0] = [0,0,0,1,1,0] means that the first engram was presented at the 4th and 5th timepoints
            engrams:[n_engrams, n_neurons_per_engrams] output of for instance get_activity_engrams

        Returns: 
            r_engrams [n_engrams (engram number), n_engram (stimulus presented number)]:  
            r_engrams[a][b] is mean activity of the neurons corresponding to engram a while stimulus b is presented.
        stimulus ordering assumed is from pretrainingeng (not from testing)
        '''
        self.get_binned_spikes_big()
        n_neurons = self.params["n_recorded"]
        n_engrams = self.params["n_tot_stim"]
        r_engrams = np.zeros( (n_engrams, n_engrams) )
        for engram_num in range(n_engrams):
            neur_ind_keep = [True if i in self.engrams[engram_num] else False for i in range(n_neurons)]
            mean_engram_rate = np.mean(self.binned_spikes_big[test_num,neur_ind_keep,:], axis=0)/self.params["bin_size_big"]
            for presented_stim_num in range(n_engrams):
                r_engrams[engram_num,presented_stim_num] = np.mean(mean_engram_rate[inds_stim[presented_stim_num]])   
        return(r_engrams)

    def compute_seq_metrics(self, resp_eng_stim):
        '''
        Assuming 5 familiar and 2 novel stimuli. 2 novel presented first then 5 familiar
        resp_eng_stim: [n_engrams (engram number), n_engram (stimulus presented number)] output of get_responses_engram_each_stim
        
        metrics [5, n_fam]: -2 -1 0 +1 +2: 5 metrics for each stimulus
            +1 metric: [r_engfam[i+1]_during_fam[i]stim)/mean(r_engnov[1]_during_fam[i]stim+r_engnov[2]_during_fam[i]stim] (shape n_fam_stim, if i+1 > n_fam then cycle to 0)
            no mean so we can compute error bars
        '''
        #maybe later write more general code for different numbers of nov and fam
        stim_label=['nov1','nov2','fam1','fam2','fam3','fam4','fam5']
        r_dict = dict()
        for i in range(7):
            r_dict[stim_label[i]] = dict()
            for j in range(7):
                r_dict[stim_label[i]]['during_'+stim_label[j]] = resp_eng_stim[i,j]
        metrics = np.zeros( (5, 5) )
        
        #-2 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam4' # during fam1 presentation look at eng_fam4 (-2)
        metrics[0,0] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam2'; label_engram = 'fam5'
        metrics[0,1] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam3'; label_engram = 'fam1'
        metrics[0,2] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam4'; label_engram = 'fam2'
        metrics[0,3] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam5'; label_engram = 'fam3'
        metrics[0,4] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        
        #-1 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam5' # during fam1 presentation look at eng_fam5 (-1)
        metrics[1,0] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam2'; label_engram = 'fam1'
        metrics[1,1] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam3'; label_engram = 'fam2'
        metrics[1,2] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam4'; label_engram = 'fam3'
        metrics[1,3] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam5'; label_engram = 'fam4'
        metrics[1,4] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        
        #0 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam1' # during fam1 presentation look at eng_fam1 (0)
        metrics[2,0] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam2'; label_engram = 'fam2'
        metrics[2,1] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam3'; label_engram = 'fam3'
        metrics[2,2] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam4'; label_engram = 'fam4'
        metrics[2,3] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam5'; label_engram = 'fam5'
        metrics[2,4] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        
        #+1 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam2' # during fam1 presentation look at eng_fam2 (+1)
        metrics[3,0] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam2'; label_engram = 'fam3'
        metrics[3,1] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam3'; label_engram = 'fam4'
        metrics[3,2] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam4'; label_engram = 'fam5'
        metrics[3,3] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam5'; label_engram = 'fam1'
        metrics[3,4] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        
        #+2 metric
        label_stim_pres = 'during_fam1'; label_engram = 'fam3' # during fam1 presentation look at eng_fam3 (+2)
        metrics[4,0] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam2'; label_engram = 'fam4'
        metrics[4,1] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam3'; label_engram = 'fam5'
        metrics[4,2] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam4'; label_engram = 'fam1'
        metrics[4,3] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        label_stim_pres = 'during_fam5'; label_engram = 'fam2'
        metrics[4,4] = r_dict[label_engram][label_stim_pres]/np.mean([r_dict['nov1'][label_stim_pres],r_dict['nov2'][label_stim_pres]])
        
        return(metrics)

    def get_pres_timepoints(self, start_test=None, on_time=None, off_time=None):
        """
        assuming 5 familiar stimuli and 2 novel.
        """
        ts = np.zeros(22) #7 stimuli presented in isolation and 3x5 sequences of stimuli
        ts[0] = start_test
        for i in range(1,8):
            ts[i] = ts[i-1] + on_time + off_time
        for i in range(8,22):
            ts[i] = ts[i-1] + 4*on_time + off_time
        return(ts)

    def get_pop_rate_1t(self,start=0,duration=1):
        n_spikes = 0
        for neuron in range(self.params["n_recorded"]):
            n_spikes += np.sum(np.logical_and(start<self.spiketimes[str(neuron)], self.spiketimes[str(neuron)]<start+duration))
        return(n_spikes/duration/self.params["n_recorded"])

    #### Stability metrics, computed just before every test session
    @property
    @_return_nan
    def rate(self):
        """
        Total population rate
        """
        r = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            start = self.test_starts[i] - self.params["l_pre_test_record"]
            stop = self.test_starts[i]
            r[i] = np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()])/self.params["l_pre_test_record"]
        return(r)
    
    @property
    @_return_nan
    def rate_i(self):
        """
        Total population rate
        """
        r = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            start = self.test_starts[i] - self.params["l_pre_test_record"]
            stop = self.test_starts[i]
            r[i] = np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes_i.values()])/self.params["l_pre_test_record"]
        return(r)
    
    @property
    @_return_nan
    def weef(self):
        """final mean EE weight"""
        return(np.mean(self.weights["ee"][:,-1]))
    
    @property
    @_return_nan
    def weif(self):
        """final mean EI weight"""
        return(np.mean(self.weights["ei"][:,-1]))
    
    @property
    @_return_nan
    def wief(self):
        """final mean IE weight"""
        return(np.mean(self.weights["ie"][:,-1]))
    
    @property
    @_return_nan
    def wiif(self):
        """final mean II weight"""
        return(np.mean(self.weights["ii"][:,-1]))

    #### Metrics from the familiarity task, computed on the first part of every test session (single stimulus presentation)
    @property
    @_return_nan
    def r_nov(self):
        """pop rate in response to novel stimuli"""
        r = np.zeros((self.n_tests, self.params["n_nov_stim"]))
        ind_pres_nov_stim = np.argwhere(['n' in i for i in self.params["ordering_test_isolation"]])
        ind_pres_nov_stim = np.reshape(ind_pres_nov_stim, ind_pres_nov_stim.shape[0])
        for i in range(self.n_tests):
            for j in range(len(ind_pres_nov_stim)):
                start = self.test_starts[i] + j*(self.params["ontime_test"] + self.params["offtime_test"])
                stop = start + self.params["ontime_test"]
                r[i,j] = np.mean([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])/self.params["ontime_test"]
        return(r)
    
    @property
    @_return_nan
    def r_fam(self):
        """pop rate in response to familiar stimuli"""
        r = np.zeros((self.n_tests, self.params["n_fam_stim"]))
        ind_pres_fam_stim = np.argwhere(['f' in i for i in self.params["ordering_test_isolation"]])
        ind_pres_fam_stim = np.reshape(ind_pres_fam_stim, ind_pres_fam_stim.shape[0])
        for i in range(self.n_tests):
            for j in range(len(ind_pres_fam_stim)):
                start = self.test_starts[i] + ind_pres_fam_stim[j]*(self.params["ontime_test"] + self.params["offtime_test"])
                stop = start + self.params["ontime_test"]
                r[i,j] = np.mean([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])/self.params["ontime_test"]
        return(r)
    
    @property
    @_return_nan
    def std_nov(self):
        """std of individual neurons rate in response to nov stimulus"""
        stdr = np.zeros((self.n_tests, self.params["n_nov_stim"]))
        ind_pres_nov_stim = np.argwhere(['n' in i for i in self.params["ordering_test_isolation"]])
        ind_pres_nov_stim = np.reshape(ind_pres_nov_stim, ind_pres_nov_stim.shape[0])
        for i in range(self.n_tests):
            for j in range(len(ind_pres_nov_stim)):
                start = self.test_starts[i] + j*(self.params["ontime_test"] + self.params["offtime_test"])
                stop = start + self.params["ontime_test"]
                stdr[i,j] = np.std([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])
        return(stdr)
    
    @property
    @_return_nan
    def std_fam(self):
        """std of individual neurons rate in response to familiar stimulus"""
        stdr = np.zeros((self.n_tests, self.params["n_fam_stim"]))
        ind_pres_fam_stim = np.argwhere(['f' in i for i in self.params["ordering_test_isolation"]])
        ind_pres_fam_stim = np.reshape(ind_pres_fam_stim, ind_pres_fam_stim.shape[0])
        for i in range(self.n_tests):
            for j in range(len(ind_pres_fam_stim)):
                start = self.test_starts[i] + j*(self.params["ontime_test"] + self.params["offtime_test"])
                stop = start + self.params["ontime_test"]
                stdr[i,j] = np.std([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])
        return(stdr)


    #### Metrics for the sequential task
#     @property
#     @_return_nan
#     def successor_rep(self):
#         """
#         numpy array [n_tests, 5, n_fam]: -2 -1 0 +1 +2: 5 metrics for each stimulus (n_fam) and for each test_phase
#         +1 metric: [r_engfam[i+1]_during_fam[i]stim)/
#                     mean(r_engnov[1]_during_fam[i]stim+r_engnov[2]_during_fam[i]stim] 
#         (shape n_fam_stim, if i+1 > n_fam then cycle to 0)
#         """
#         self.get_binned_spikes_big()
#         succ = np.zeros((self.n_tests,5,self.params["n_fam_stim"]))
#         for test_num in range(self.n_tests):
#             inds_stim = self.get_inds_single_stim_pres(test_num)
#             resp_eng_stim = self.get_responses_engram_each_stim(inds_stim, test_num)
#             succ[test_num] = self.compute_seq_metrics(resp_eng_stim)
#         return(succ)
    
    @property
    @_return_nan
    def prate(self):
        self.get_binned_spikes_big()
        return(np.mean(self.binned_spikes_big, axis=1)/self.params["bin_size_big"])
    
    
    @property
    @_return_nan
    def eng_rate(self):
        self.get_binned_spikes_big()
        eng_rate = np.zeros((self.n_tests, self.params["n_tot_stim"], self.binned_spikes_big.shape[2]))
        for engram_num in range(self.params["n_tot_stim"]):
            neur_ind_keep = [True if i in self.engrams[engram_num] else False for i in range(self.params["n_recorded"])]
            eng_rate[:,engram_num, :] = np.mean(self.binned_spikes_big[:,neur_ind_keep,:], axis=1)
        return(eng_rate/self.params["bin_size_big"])
    
    @property
    @_return_nan
    def non_eng_rate(self):
        ### shape [self.n_tests, self.binned_spikes_big.shape[2]]
        self.get_binned_spikes_big()
        neur_ind_fameng = np.array([False for i in range(self.params["n_recorded"])], dtype=bool)
        for fam_engram_num in range(2,self.params["n_tot_stim"]): #select all engram neurons associated to fam1,2,3,4,5 
            neur_ind_fameng = np.logical_or( neur_ind_fameng, np.array([True if i in self.engrams[fam_engram_num] else False for i in range(self.params["n_recorded"])], dtype=bool) )
        return( np.mean(self.binned_spikes_big[:,np.logical_not(neur_ind_fameng),:], axis=1)/self.params["bin_size_big"] )
    



class ComputeMetrics_BND:
    """Compute metrics given a simulation of the novelty detection task exclusively
    For now nothing on the weights, se the old version of compute metrics for that."""

    def __init__(self, spiketimes: dict, 
                 sim_params: dict, 
                 hard_coded_sim_params: dict,
                 weights: dict=None, 
                 spiketimes_i: dict=None) -> None:
        """Set up class to compute metrics."""
        # initialise an object directly with spiketimes dict, weight dict and
        # params dict
        self.spiketimes = spiketimes
        self.spiketimes_i = spiketimes_i
        self.params = sim_params
        for key in hard_coded_sim_params.keys():
            self.params[key] = hard_coded_sim_params[key]

        self.weights = weights

        self.binned_spikes_big_computed = False
        self.binned_pop_rate_computed = False
        self.isis_computed = False
        self.cvs_computed = False
        
        self.get_test_phase_start_times()
        self.n_tests = len(self.params["break_durations"])

    def _check(self, keys):
        assert np.all([k in self.params.keys() for k in keys])

    def _return_nan(metric_func):
        def modify_metric_to_return_nan(self):
            if self.spiketimes is None or self.weights is None:
                return np.nan
            else:
                return metric_func(self)
        return modify_metric_to_return_nan

    def get_individual_rates(self, start, stop):
        """
        Compute the the firing of each recorded (excitatory) neuron 
        between start and stop
        """
        self._check(["n_recorded"])
        rates = np.zeros(self.params["n_recorded"])
        for neuron in range(self.params["n_recorded"]):
            rates[neuron] =  np.sum(np.logical_and(start<=self.spiketimes[str(neuron)], 
                                                   self.spiketimes[str(neuron)]<=stop))/(stop-start)
        return(rates)
    
    def get_test_phase_start_times(self):
        """
        returns the starting time in simulated time (presentation time of the novel stimulus), does not include the 1s of recording before test starts (self.params["l_pre_test_record"]).
        """
        self.l_1test = 2*(self.params["ontime_test"] + self.params["offtime_test"])
        self.test_starts = np.zeros(len(self.params["break_durations"]))
        self.test_starts[0] = self.params["lpt"] + self.params["lt"] + self.params["break_durations"][0]
        for i in range(1,len(self.params["break_durations"])):
            self.test_starts[i] = self.test_starts[i-1] + self.l_1test + self.params["break_durations"][i]

    def get_binned_spikes_big(self):
        """
        computes binned_spikes for each recorded neuron. 
        Since we have several test sessions during a single simulation, we use a type
        ts = np.array[n_test_phases, n_bins]
        binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
        """
        if not self.binned_spikes_big_computed: 
            n_bins_per_test = len(np.arange(self.test_starts[0]-self.params["l_pre_test_record"], self.test_starts[0]+self.l_1test, self.params["bin_size_big"]))-1
            self.ts_big = np.zeros((self.n_tests, n_bins_per_test))
            self.binned_spikes_big = np.zeros((self.n_tests, self.params["n_recorded"], n_bins_per_test))
            for i in range(self.n_tests):
                bins = np.arange(self.test_starts[i]-self.params["l_pre_test_record"], self.test_starts[i]+self.l_1test, self.params["bin_size_big"])
                self.binned_spikes_big[i] = np.array([np.histogram(self.spiketimes[str(neuron_num)], bins=bins)[0] \
                        for neuron_num in range(self.params["n_recorded"])])
                self.ts_big[i] = bins[:-1]
            self.binned_spikes_big_computed = True

    def get_binned_pop_rate(self):
        """
        computes the population firing rate for all test sessions with bin_size_big
        ts = np.array[n_test_phases, n_bins]
        binned_spikes = np.array[n_test_phases, n_neurons, n_bins]
        """   
        if not self.binned_pop_rate_computed: 
            self.get_binned_spikes_big()
            n_bins_per_test = len(np.arange(self.test_starts[0]-self.params["l_pre_test_record"], self.test_starts[0]+self.l_1test, self.params["bin_size_big"]))-1
            self.binned_pop_rate = np.zeros((self.n_tests, n_bins_per_test))
            for i in range(self.n_tests):
                bins = np.arange(self.test_starts[i]-self.params["l_pre_test_record"], self.test_starts[i]+self.l_1test, self.params["bin_size_big"])
                self.binned_pop_rate[i] = np.mean(self.binned_spikes_big[i], axis=0)
            self.binned_pop_rate = self.binned_pop_rate/self.params["bin_size_big"]
            self.binned_pop_rate_computed = True

    def get_pop_rate_1t(self,start=0,duration=1):
        n_spikes = 0
        for neuron in range(self.params["n_recorded"]):
            n_spikes += np.sum(np.logical_and(start<self.spiketimes[str(neuron)], self.spiketimes[str(neuron)]<start+duration))
        return(n_spikes/duration/self.params["n_recorded"])
    
    def get_isis(self):
        if not self.isis_computed:
            self.isis = dict()
            for i in range(self.n_tests):
                self.isis[str(i)] = dict()
                t_start = self.test_starts[i] - 0.001
                t_stop = self.test_starts[i] + 2*(self.params["ontime_test"] + self.params["offtime_test"]) + 0.001
                for neuron_num in range(self.params["n_recorded"]):
                    inds_to_keep = np.argwhere( np.logical_and(t_start<=self.spiketimes[str(neuron_num)],
                                                               self.spiketimes[str(neuron_num)]<=t_stop) )
                    self.isis[str(i)][str(neuron_num)] = np.diff(self.spiketimes[str(neuron_num)][inds_to_keep][:,0])
            self.isis_computed = True
             
        return(self.isis)

    def get_cvs(self):
        self.get_isis()
        if not self.cvs_computed:
            self.cvs = dict()
            for i in range(self.n_tests):
                self.cvs[str(i)] = np.zeros(self.params["n_recorded"])
                for neuron_num in range(self.params["n_recorded"]):
                    if len(self.isis[str(i)][str(neuron_num)]) > 2:
                        self.cvs[str(i)][neuron_num] = np.std(self.isis[str(i)][str(neuron_num)]) / np.mean(self.isis[str(i)][str(neuron_num)])
            self.cvs_computed = True
        return(self.cvs)

    #### Stability metrics, computed just before every test session
    @property
    @_return_nan
    def rate(self):
        """
        Total population rate
        """
        r = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            start = self.test_starts[i] - self.params["l_pre_test_record"]
            stop = self.test_starts[i]
            r[i] = np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes.values()])/self.params["l_pre_test_record"]
        return(r)
    
    @property
    @_return_nan
    def rate_i(self):
        """
        Total population rate
        """
        r = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            start = self.test_starts[i] - self.params["l_pre_test_record"]
            stop = self.test_starts[i]
            r[i] = np.mean([np.sum(np.logical_and(start<=i, i<=stop)) for i in self.spiketimes_i.values()])/self.params["l_pre_test_record"]
        return(r)

    @property
    @_return_nan
    def r_nov(self):
        """pop rate in response to novel stimuli"""
        r = np.zeros((self.n_tests))
        for i in range(self.n_tests):
            start = self.test_starts[i]
            stop = start + self.params["ontime_test"]
            r[i] = np.mean([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])/self.params["ontime_test"]
        return(r)
    
    @property
    @_return_nan
    def r_fam(self):
        """pop rate in response to familiar stimuli"""
        r = np.zeros((self.n_tests))
        for i in range(self.n_tests):
            start = self.test_starts[i] + self.params["ontime_test"] + self.params["offtime_test"]
            stop = start + self.params["ontime_test"]
            r[i] = np.mean([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])/self.params["ontime_test"]
        return(r)
    
    @property
    @_return_nan
    def std_nov(self):
        """std of individual neurons rate in response to nov stimulus"""
        stdr = np.zeros((self.n_tests))
        for i in range(self.n_tests):
            start = self.test_starts[i]
            stop = start + self.params["ontime_test"]
            stdr[i] = np.std([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])
        return(stdr)
    
    @property
    @_return_nan
    def std_fam(self):
        """std of individual neurons rate in response to familiar stimulus"""
        stdr = np.zeros((self.n_tests))
        for i in range(self.n_tests):
            start = self.test_starts[i] + self.params["ontime_test"] + self.params["offtime_test"]
            stop = start + self.params["ontime_test"]
            stdr[i] = np.std([np.sum(np.logical_and(start<=k, k<=stop)) for k in self.spiketimes.values()])
        return(stdr)
    
    @property
    @_return_nan
    def prate(self):
        self.get_binned_spikes_big()
        return(np.mean(self.binned_spikes_big, axis=1)/self.params["bin_size_big"])
    
    @property
    @_return_nan
    def weef(self):
        """initial mean EE weight"""
        wee = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            ind_begin_test = np.argwhere( np.logical_and(self.test_starts[i]-0.001<=self.weights["t"], self.weights["t"]<=self.test_starts[i]+0.001) )[0,0]
            wee[i] = np.mean(self.weights["ee"][:,ind_begin_test])
        return(wee)
    
    @property
    @_return_nan
    def weif(self):
        """initial mean EI weight"""
        wei = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            ind_begin_test = np.argwhere( np.logical_and(self.test_starts[i]-0.001<=self.weights["t"], self.weights["t"]<=self.test_starts[i]+0.001) )[0,0]
            wei[i] = np.mean(self.weights["ei"][:,ind_begin_test])
        return(wei)
    
    @property
    @_return_nan
    def wief(self):
        """initial mean IE weight"""
        wie = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            ind_begin_test = np.argwhere( np.logical_and(self.test_starts[i]-0.001<=self.weights["t"], self.weights["t"]<=self.test_starts[i]+0.001) )[0,0]
            wie[i] = np.mean(self.weights["ie"][:,ind_begin_test])
        return(wie)
    
    
    @property
    @_return_nan
    def wiif(self):
        """initial mean II weight"""
        wii = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            ind_begin_test = np.argwhere( np.logical_and(self.test_starts[i]-0.001<=self.weights["t"], self.weights["t"]<=self.test_starts[i]+0.001) )[0,0]
            wii[i] = np.mean(self.weights["ii"][:,ind_begin_test])
        return(wii)
    
    @property
    @_return_nan
    def w_blow(self):
        """Indicate if synaptic weights have exploded."""
        # check that the simulation has the params for the metric to be computed
        self._check(["wmax"])
        
        wblow = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            f_blow = 0
            for key in self.weights.keys():
                if key != "t":
                    t_start = self.test_starts[i] - 0.001
                    t_stop = self.test_starts[i] + 2*(self.params["ontime_test"] + self.params["offtime_test"]) + 0.001
                    w_distr = get_w_distr(w_dict={"w": self.weights[key], "t": self.weights["t"]},
                                          t_start=t_start, t_stop=t_stop)
                    f_blow += np.sum([i == 0 or i == self.params["wmax"] for i in w_distr]) / len(w_distr)
            wblow[i] = f_blow / (len(self.weights.keys()) - 1)
            
        return(wblow)
    
    @property
    @_return_nan
    def w_creep(self):
        """compute change of mean weight between start and finish (as percentage), max amount all weights considered"""
        wc = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            w_creep_metric = 0
            
            t_start = self.test_starts[i]
            ind_start = np.argwhere( np.logical_and(t_start-0.001<=self.weights["t"], self.weights["t"]<=t_start+0.001) )[0,0]
            #-0.1 because in c++ record frequency 0.1s, and we don't have the last bin. aka if t_end = 20 last t in w is 19.9
            t_end = self.test_starts[i] + 2*(self.params["ontime_test"] + self.params["offtime_test"]) - 0.1 
            ind_end = np.argwhere( np.logical_and(t_end-0.001<=self.weights["t"], self.weights["t"]<=t_end+0.001) )[0,0]
            
            for key in self.weights.keys():
                if key != "t":
                    start_w = np.mean(self.weights[key][:,ind_start])
                    end_w = np.mean(self.weights[key][:,ind_end])
                    if start_w + end_w > 0.1:
                        candidate = np.abs(2*(end_w - start_w)/(end_w + start_w))
                        if candidate > w_creep_metric:
                            w_creep_metric = candidate
            wc[i] = w_creep_metric
        return(wc)
    
    @property
    @_return_nan
    def cv_isi(self):
        """Coefficient of variation of exc neurons' interspike-interval distribution, averaged over neurons"""
        cv = np.zeros(self.n_tests)
        for i in range(self.n_tests):
            cv[i] = np.mean(self.get_cvs()[str(i)])
        return(cv)
    

    

    

    
