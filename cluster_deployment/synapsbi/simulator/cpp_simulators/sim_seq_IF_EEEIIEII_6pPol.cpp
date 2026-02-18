#include "auryn.h"
#include "SixParamConnection.h"
#include "RandStimGroup.h"
#include "RFConnectionSeq.h"
#include "WeightMonitor_wSwitch.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>

namespace po = boost::program_options;
using namespace auryn;


std::vector<float> parse_input_plasticity(std::vector<std::string> rule_str)
{
	// parsing the command line rule argument: (it is given as a string because multitoken() is bugged: negative numbers cause errors)
	std::vector<float> rule(6);
	std::string s = rule_str[0];
	std::string delimiter = "a";
	size_t pos = 0;
	int ct = 0;
	std::string token;

	token = s.substr(0, pos); // remove the first a (needed in case first param is negative, or is it?)
	s.erase(0, pos + delimiter.length());

	while ((pos = s.find(delimiter)) != std::string::npos) { // parse the rest of the expression
		token = s.substr(0, pos);
		rule[ct] = boost::lexical_cast<float>(token);
		s.erase(0, pos + delimiter.length());
		ct ++;
	}
	return(rule);
}

// until 04/11/2024, for 7056 inputs
// std::vector< std::vector< std::vector<float> > > PatternGenerator(int pat_num, int N_inputs)
// {
//     int N_inp_2D = (int) std::sqrt(N_inputs);
//     int N_inp_2D_1_3 = (int) N_inp_2D/3;
//     int N_inp_2D_1_7 = (int) N_inp_2D/7;
//     std::vector< std::vector< std::vector<float> > > pat_array(1, std::vector< std::vector<float> >( (int) N_inputs/7 , std::vector<float>(2)));
//     int start_1st_col, start_2nd_col, start_3rd_col;
//     if (pat_num == 0){start_1st_col = 0; start_2nd_col = 6; start_3rd_col = 2;}
//     if (pat_num == 1){start_1st_col = 1; start_2nd_col = 2; start_3rd_col = 5;}
//     if (pat_num == 2){start_1st_col = 2; start_2nd_col = 4; start_3rd_col = 3;}
//     if (pat_num == 3){start_1st_col = 3; start_2nd_col = 1; start_3rd_col = 1;}
//     if (pat_num == 4){start_1st_col = 4; start_2nd_col = 5; start_3rd_col = 6;}
//     if (pat_num == 5){start_1st_col = 5; start_2nd_col = 0; start_3rd_col = 4;}
//     if (pat_num == 6){start_1st_col = 6; start_2nd_col = 3; start_3rd_col = 0;}
    
//     // add neurons from the first column in input space
//     int counter = 0;
//     for (int i = start_1st_col*N_inp_2D_1_7; i < (start_1st_col+1)*N_inp_2D_1_7; i++){
//         for (int j = 0; j < N_inp_2D_1_3; j++){
//             pat_array[0][counter][0] = i*N_inp_2D+j;
//             pat_array[0][counter][1] = 1;
//             counter ++;
//         }
//     }
//     // add neurons from the second column in input space
//     for (int i = start_2nd_col*N_inp_2D_1_7; i < (start_2nd_col+1)*N_inp_2D_1_7; i++){
//         for (int j = N_inp_2D_1_3; j < 2*N_inp_2D_1_3; j++){
//             pat_array[0][counter][0] = i*N_inp_2D+j;
//             pat_array[0][counter][1] = 1;
//             counter ++;
//         }
//     }
//     // add neurons from the third column in input space
//     for (int i = start_3rd_col*N_inp_2D_1_7; i < (start_3rd_col+1)*N_inp_2D_1_7; i++){
//         for (int j = 2*N_inp_2D_1_3; j < 3*N_inp_2D_1_3; j++){
//             pat_array[0][counter][0] = i*N_inp_2D+j;
//             pat_array[0][counter][1] = 1;
//             counter ++;
//         }
//     }
//     return pat_array;
// }

// post 04/11/2024: added a 7-wide padding on the vertical 3 columns, so that there are no small neighbours anymore.
std::vector< std::vector< std::vector<float> > > PatternGenerator(int pat_num, int N_inputs)
{
    int N_inp_2D = (int) std::sqrt(N_inputs);
    int N_inp_2D_1_3 = (int) N_inp_2D/3;
    int N_inp_2D_1_7 = (int) N_inp_2D/7;
    std::vector< std::vector< std::vector<float> > > pat_array(1, std::vector< std::vector<float> >( (int) N_inputs/7 , std::vector<float>(2)));
    int start_1st_col, start_2nd_col, start_3rd_col;
    if (pat_num == 0){start_1st_col = 0; start_2nd_col = 6; start_3rd_col = 2;}
    if (pat_num == 1){start_1st_col = 1; start_2nd_col = 2; start_3rd_col = 5;}
    if (pat_num == 2){start_1st_col = 2; start_2nd_col = 4; start_3rd_col = 3;}
    if (pat_num == 3){start_1st_col = 3; start_2nd_col = 1; start_3rd_col = 1;}
    if (pat_num == 4){start_1st_col = 4; start_2nd_col = 5; start_3rd_col = 6;}
    if (pat_num == 5){start_1st_col = 5; start_2nd_col = 0; start_3rd_col = 4;}
    if (pat_num == 6){start_1st_col = 6; start_2nd_col = 3; start_3rd_col = 0;}
    
    // add neurons from the first column in input space
    int counter = 0;
    for (int i = start_1st_col*N_inp_2D_1_7; i < (start_1st_col+1)*N_inp_2D_1_7; i++){
        for (int j = 5; j < N_inp_2D_1_3 - 5; j++){
            pat_array[0][counter][0] = i*N_inp_2D+j;
            pat_array[0][counter][1] = 1;
            counter ++;
        }
    }
    // add neurons from the second column in input space
    for (int i = start_2nd_col*N_inp_2D_1_7; i < (start_2nd_col+1)*N_inp_2D_1_7; i++){
        for (int j = N_inp_2D_1_3 + 5; j < 2*N_inp_2D_1_3 - 5; j++){
            pat_array[0][counter][0] = i*N_inp_2D+j;
            pat_array[0][counter][1] = 1;
            counter ++;
        }
    }
    // add neurons from the third column in input space
    for (int i = start_3rd_col*N_inp_2D_1_7; i < (start_3rd_col+1)*N_inp_2D_1_7; i++){
        for (int j = 2*N_inp_2D_1_3 + 5; j < 3*N_inp_2D_1_3 - 5; j++){
            pat_array[0][counter][0] = i*N_inp_2D+j;
            pat_array[0][counter][1] = 1;
            counter ++;
        }
    }
    return pat_array;
}


int main(int ac, char* av[])
{
	/////////////////////////////////////////////////////////
	// Get simulation parameters from command line options //
	/////////////////////////////////////////////////////////

	std::string ID = "0";
	std::string workdir;

	float wee = 0.;
	float wei = 0.;
	float wie = 0.;
	float wii = 0.;
	float sparseness = 0.;

	float rate_poisson = 0.;
	float weight_poisson = 0.;
	int radius = 0;
	int N_active_input = 0;
	float active_input_rate = 0.;

	std::vector<float> ruleEE(6);
	std::vector<std::string> ruleEE_str;
	std::vector<float> ruleEI(6);
	std::vector<std::string> ruleEI_str;
	std::vector<float> ruleIE(6);
	std::vector<std::string> ruleIE_str;
	std::vector<float> ruleII(6);
	std::vector<std::string> ruleII_str;
	float eta = 0.;
	float wmax = 0.;

	double l_pre_train;
	double l_train;
	double l_stim_on_train;
	double l_stim_off_train;
	double l_stim_on_test;
	double l_stim_off_test;

	int NE = 0;
	int NI = 0;
	int N_inputs = 0;
	float tau_ampa = 0;
	float tau_gaba = 0;
	float tau_nmda = 0;
	float ampa_nmda_ratio = 0;

	float max_rate_checker = 0.;
	float tau_checker = 0.;

	int n_recorded = 100;
	bool record_i = false;
	int n_recorded_i = 500;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
			("ID", po::value<std:: string>(), "ID to name the monitor output files correctly")
			("NE", po::value<int>(), "NE")
			("NI", po::value<int>(), "NI")
			("tau_ampa", po::value<float>(), "tau_ampa")
			("tau_gaba", po::value<float>(), "tau_gaba")
			("tau_nmda", po::value<float>(), "tau_nmda")
			("ampa_nmda_ratio", po::value<float>(), "ampa_nmda_ratio")
			("ruleEE", po::value< std::vector<std::string> >(), "plasticity rule for EE, to enter as a string with separator a (start with a)")
			("ruleEI", po::value< std::vector<std::string> >(), "plasticity rule for EI")
			("ruleIE", po::value< std::vector<std::string> >(), "plasticity rule for IE")
			("ruleII", po::value< std::vector<std::string> >(), "plasticity rule for II")
			("eta", po::value<float>(), "lerning rate for the rule")
			("wmax", po::value<float>(), "max exc weight")
			("wee", po::value<float>(), "wee")
			("wei", po::value<float>(), "wei")
			("wie", po::value<float>(), "wie")
			("wii", po::value<float>(), "wii")
			("sparseness", po::value<float>(), "sparseness")
			("N_inputs", po::value<int>(), "N_inputs")
			("rate_poisson", po::value<float>(), "rate_poisson")
			("weight_poisson", po::value<float>(), "weight_poisson")
			("radius", po::value<int>(), "radius")
			("active_input_rate", po::value<float>(), "active_input_rate")
			("ontime_train", po::value<double>(), "ontime_train")
			("offtime_train", po::value<double>(), "offtime_train")
			("ontime_test", po::value<double>(), "ontime_test")
			("offtime_test", po::value<double>(), "offtime_test")
			("max_rate_checker", po::value<float>(), "max_rate_checker")
			("tau_checker", po::value<float>(), "tau_checker")
			("lpt", po::value<double>(), "length_pre_train")
			("lt", po::value<double>(), "length_train")
			("workdir", po::value<std::string>(), "workdir to write output files (until we have a writeless monitor)")
			("n_recorded", po::value<int>(), "how many exc neurons to record")
			("record_i", po::value<bool>(), " whetyher to record inhibitory spikes or not")
			("n_recorded_i", po::value<int>(), "how many inh neurons to record, relevant only if record_i is true")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);

		if (vm.count("ID")) {ID= vm["ID"].as<std::string>();}
		if (vm.count("NE")) {NE= vm["NE"].as<int>();}
		if (vm.count("NI")) {NI= vm["NI"].as<int>();}
		if (vm.count("tau_ampa")) {tau_ampa = vm["tau_ampa"].as<float>();}
		if (vm.count("tau_gaba")) {tau_gaba = vm["tau_gaba"].as<float>();}
		if (vm.count("tau_nmda")) {tau_nmda = vm["tau_nmda"].as<float>();}
		if (vm.count("ampa_nmda_ratio")) {ampa_nmda_ratio = vm["ampa_nmda_ratio"].as<float>();}
		if (vm.count("ruleEE")) {ruleEE_str = vm["ruleEE"].as< std::vector<std::string> >();}
		if (vm.count("ruleEI")) {ruleEI_str = vm["ruleEI"].as< std::vector<std::string> >();}
		if (vm.count("ruleIE")) {ruleIE_str = vm["ruleIE"].as< std::vector<std::string> >();}
		if (vm.count("ruleII")) {ruleII_str = vm["ruleII"].as< std::vector<std::string> >();}
		if (vm.count("eta")) {eta = vm["eta"].as<float>();}
		if (vm.count("wmax")) {wmax = vm["wmax"].as<float>();}
		if (vm.count("wee")) {wee = vm["wee"].as<float>();}
		if (vm.count("wei")) {wei = vm["wei"].as<float>();}
		if (vm.count("wie")) {wie = vm["wie"].as<float>();}
		if (vm.count("wii")) {wii = vm["wii"].as<float>();}
		if (vm.count("sparseness")) {sparseness = vm["sparseness"].as<float>();}
		if (vm.count("N_inputs")) {N_inputs= vm["N_inputs"].as<int>();}
		if (vm.count("rate_poisson")) {rate_poisson = vm["rate_poisson"].as<float>();}
		if (vm.count("weight_poisson")) {weight_poisson = vm["weight_poisson"].as<float>();}
		if (vm.count("radius")) {radius = vm["radius"].as<int>();}
		if (vm.count("active_input_rate")) {active_input_rate = vm["active_input_rate"].as<float>();}
		if (vm.count("ontime_train")) {l_stim_on_train = vm["ontime_train"].as<double>();}
		if (vm.count("offtime_train")) {l_stim_off_train = vm["offtime_train"].as<double>();}
		if (vm.count("ontime_test")) {l_stim_on_test = vm["ontime_test"].as<double>();}
		if (vm.count("offtime_test")) {l_stim_off_test = vm["offtime_test"].as<double>();}
		if (vm.count("max_rate_checker")) {max_rate_checker = vm["max_rate_checker"].as<float>();}
		if (vm.count("tau_checker")) {tau_checker = vm["tau_checker"].as<float>();}
		if (vm.count("lpt")) {l_pre_train = vm["lpt"].as<double>();}
		if (vm.count("lt")) {l_train = vm["lt"].as<double>();}
		if (vm.count("workdir")) {workdir = vm["workdir"].as<std::string>();}
		if (vm.count("n_recorded")) {n_recorded = vm["n_recorded"].as<int>();}
		if (vm.count("record_i")) {record_i= vm["record_i"].as<bool>();}
		if (vm.count("n_recorded_i")) {n_recorded_i= vm["n_recorded_i"].as<int>();}
	}
	catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	// parsing the command line rule arguments
	ruleEE = parse_input_plasticity(ruleEE_str);
	ruleEI = parse_input_plasticity(ruleEI_str);
	ruleIE = parse_input_plasticity(ruleIE_str);
	ruleII = parse_input_plasticity(ruleII_str);

	///////////////////////
	// Build the network //
	///////////////////////

	auryn_init(ac, av, workdir.c_str(), "default", "", NONE, NONE);
	sys->quiet = true;

	// handle randomness of simulation: by default random seed
	std::srand(std::time(0));
	sys->set_master_seed(std::rand());

	IFGroup* neurons_e = new IFGroup(NE);
	neurons_e->set_tau_ampa(tau_ampa); //5e-3
	neurons_e->set_tau_gaba(tau_gaba); //10e-3
	neurons_e->set_tau_nmda(tau_nmda); //100e-3
	neurons_e->set_ampa_nmda_ratio(ampa_nmda_ratio); //0.3

	IFGroup* neurons_i = new IFGroup(NI);
	neurons_i->set_tau_ampa(tau_ampa); //5e-3
	neurons_i->set_tau_gaba(tau_gaba); //10e-3
	neurons_i->set_tau_nmda(tau_nmda); //100e-3
	neurons_i->set_ampa_nmda_ratio(ampa_nmda_ratio); //0.3

	// Checker scorer to stop simulations that exceed certain firing rates
	RateChecker* cs = new RateChecker(neurons_e, 0, max_rate_checker, tau_checker);

	// External inputs to the neurons. New design 29/10/2024 to trigger balanced network responses in static nets
	std::vector< std::vector< std::vector<float> > > fam1 = PatternGenerator(0, N_inputs);
	std::vector< std::vector< std::vector<float> > > fam2 = PatternGenerator(1, N_inputs);
	std::vector< std::vector< std::vector<float> > > fam3 = PatternGenerator(2, N_inputs);
	std::vector< std::vector< std::vector<float> > > fam4 = PatternGenerator(3, N_inputs);
	std::vector< std::vector< std::vector<float> > > fam5 = PatternGenerator(4, N_inputs);
    std::vector< std::vector< std::vector<float> > > nov1 = PatternGenerator(5, N_inputs);
	std::vector< std::vector< std::vector<float> > > nov2 = PatternGenerator(6, N_inputs);

	float mean_on = 1;
	float mean_off = 0;
	RandStimGroup* stimgroup = new RandStimGroup(N_inputs, RANDOM, rate_poisson);
    stimgroup->set_mean_on_period(mean_on);
    stimgroup->set_mean_off_period(mean_off);
	stimgroup->scale = active_input_rate - rate_poisson;
    stimgroup->background_rate = rate_poisson;
    stimgroup->background_during_stimulus = true;

	PoissonGroup* poisson = new PoissonGroup(4096, rate_poisson); // no need to simulate 35k neurons for this

	RFConnectionSeq* rf_con = new RFConnectionSeq(stimgroup, neurons_e, weight_poisson, radius, GLUT);
	RFConnectionSeq* rf_con_i = new RFConnectionSeq(poisson, neurons_i, weight_poisson, radius, GLUT);

	// recurrent connectivity
	SixParamConnection* con_ee = new SixParamConnection(neurons_e,neurons_e,wee,sparseness,eta,
														ruleEE[2],ruleEE[3],ruleEE[4],ruleEE[5],ruleEE[0],ruleEE[1],wmax,GLUT);
	SixParamConnection* con_ei = new SixParamConnection(neurons_e,neurons_i,wei,sparseness,eta,
														ruleEI[2],ruleEI[3],ruleEI[4],ruleEI[5],ruleEI[0],ruleEI[1],wmax,GLUT);
	SixParamConnection* con_ie = new SixParamConnection(neurons_i,neurons_e,wie,sparseness,eta,
														ruleIE[2],ruleIE[3],ruleIE[4],ruleIE[5],ruleIE[0],ruleIE[1],wmax,GABA);
	SixParamConnection* con_ii = new SixParamConnection(neurons_i,neurons_i,wii,sparseness,eta,
														ruleII[2],ruleII[3],ruleII[4],ruleII[5],ruleII[0],ruleII[1],wmax,GABA);


	//////////////////////////////////
	/////// RUNNING THE NETWORK //////
	//////////////////////////////////

	//////////////////////////
	/// Pre-training phase ///
	//////////////////////////

	//// FOR WORKSTATION ONLY, TO TURN OFF FOR CLUSTER //////////////////////////////////
	// SpikeMonitor* smon_input = new SpikeMonitor(stimgroup , sys->fn("out.input." + ID, "ras"));
	// SpikeMonitor* smon_e = new SpikeMonitor(neurons_e , sys->fn("out.e." + ID, "ras"));
	// SpikeMonitor* smon_i = new SpikeMonitor(neurons_i , sys->fn("out.i." + ID, "ras"));
	// BinarySpikeMonitor* smon_input = new BinarySpikeMonitor(stimgroup , sys->fn("out.input."+ID,"spk"));
	// BinarySpikeMonitor* smon_e = new BinarySpikeMonitor(neurons_e , sys->fn("out.e."+ ID,"spk"));
	// BinarySpikeMonitor* smon_i = new BinarySpikeMonitor(neurons_i, sys->fn("out.i."+ ID,"spk"));
	// WeightMonitor_wSwitch* wmon_ee = new WeightMonitor_wSwitch(con_ee, sys->fn("con_ee." + ID,"syn"), 0.1);
	// wmon_ee->add_equally_spaced(1000);
	// WeightMonitor_wSwitch* wmon_ei = new WeightMonitor_wSwitch(con_ei, sys->fn("con_ei." + ID,"syn"), 0.1);
	// wmon_ei->add_equally_spaced(1000);
	// WeightMonitor_wSwitch* wmon_ie = new WeightMonitor_wSwitch(con_ie, sys->fn("con_ie." + ID,"syn"), 0.1);
	// wmon_ie->add_equally_spaced(1000);
	// WeightMonitor_wSwitch* wmon_ii = new WeightMonitor_wSwitch(con_ii, sys->fn("con_ii." + ID,"syn"), 0.1);
	// wmon_ii->add_equally_spaced(1000);
	///////////////////////////////////////////////////////////////////////////////
    
	stimgroup->clear_patterns();
	sys->run(l_pre_train);


	//////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////// Pre training assessment of engrams ///////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////

	//// TO TURN ON FOR CLUSTER //////////////////////////////////
	SpikeMonitor* smon_e = new SpikeMonitor(neurons_e , sys->fn("out.e." + ID, "ras"), n_recorded);
	SpikeMonitor* smon_i = new SpikeMonitor(neurons_i , sys->fn("out.i." + ID, "ras"), n_recorded_i);
	WeightMonitor_wSwitch* wmon_ee = new WeightMonitor_wSwitch(con_ee, sys->fn("con_ee." + ID,"syn"), 0.1);
	wmon_ee->add_equally_spaced(100);
	WeightMonitor_wSwitch* wmon_ei = new WeightMonitor_wSwitch(con_ei, sys->fn("con_ei." + ID,"syn"), 0.1);
	wmon_ei->add_equally_spaced(100);
	WeightMonitor_wSwitch* wmon_ie = new WeightMonitor_wSwitch(con_ie, sys->fn("con_ie." + ID,"syn"), 0.1);
	wmon_ie->add_equally_spaced(100);
	WeightMonitor_wSwitch* wmon_ii = new WeightMonitor_wSwitch(con_ii, sys->fn("con_ii." + ID,"syn"), 0.1);
	wmon_ii->add_equally_spaced(100);
	//////////////////////////////////////////////////////////////

	// turn off plasticity for that engram assessment
	con_ee->stdp_active = false; con_ei->stdp_active = false; con_ie->stdp_active = false; con_ii->stdp_active = false;

	float l_stim_on_pretrain = 1;
	float l_stim_off_pretrain = 1;

	sys->save_network_state("net" + ID); // we will revert to that network state for each individual stimulus presentation

	stimgroup->clear_patterns(); stimgroup->load_patterns(nov1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(nov2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID);
	stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_pretrain);
	stimgroup->clear_patterns(); sys->run(l_stim_off_pretrain);

	sys->load_network_state(workdir + "/net" + ID); // revert to network state before individual stimulus presentation

	//// TO TURN ON FOR CLUSTER //////////////////////////////////
	// turn monitors off
	smon_e->active=false; smon_i->active=false; wmon_ee->turn_off(); wmon_ei->turn_off(); wmon_ie->turn_off(); wmon_ii->turn_off();
	//////////////////////////////////////////////////////////////

	/// turn plasticity back on
	con_ee->stdp_active = true; con_ei->stdp_active = true; con_ie->stdp_active = true; con_ii->stdp_active = true;


	//////////////////////////////////////////////////////////////////////
	/////////////////////////// Training phase ///////////////////////////
	//////////////////////////////////////////////////////////////////////
    
	float t_past = 0;
	while (t_past + 5*l_stim_on_train <= l_train){ // only check if we have time for another full stim presentation, exclude the off time on purpose
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_train);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_train);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_train);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_train);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_train);
		
		t_past = t_past + 5*l_stim_on_train;
		if (t_past + l_stim_off_train <= l_train){ //do we have time to run a full break?
			stimgroup->clear_patterns(); sys->run(l_stim_off_train);
		}
		else{
			stimgroup->clear_patterns(); sys->run(l_train - t_past);
		}
		t_past = t_past + l_stim_off_train;
	}

	if (t_past < l_train){ //any leftover training time to finish
		sys->run(l_train - t_past);
	}

	//////////////////////////////////////////////////////////////////////////
	/////////////////////////// Break + Test loops ///////////////////////////
	////////////////////////////////////////////////////////////////////////// 

	const int n_breaks = 10;
// 	///////////////////// Total time: 1, 10, 20, 60, 120, 300, 600, 1200, 3600, 14400 ///
	int break_durations [n_breaks] = {1, 9,  10, 40, 60,  180, 300, 600,  2400, 10800};
    
	// const int n_breaks = 2;
	///////////////////// Total time: 1, 10, 20, 60, 120 ///
	// int break_durations [n_breaks] = {1, 9};

	for (int i=0; i<n_breaks; i++){ 

		////////////////////
		/// Break period ///
		////////////////////

		/// Break period with no recording
		//// TO TURN ON FOR CLUSTER //////////////////////////////////
		smon_e->active=false; smon_i->active=false; wmon_ee->turn_off(); wmon_ei->turn_off(); wmon_ie->turn_off(); wmon_ii->turn_off();
		/////////////////////////////////////////////////////////////
		stimgroup->clear_patterns(); sys->run(break_durations[i]-1);

		/// Turn on monitors for final second of break and test
		//// TO TURN ON FOR CLUSTER //////////////////////////////////
		smon_e->active=true; smon_i->active=true; wmon_ee->turn_on(); wmon_ei->turn_on(); wmon_ie->turn_on(); wmon_ii->turn_on();
		/////////////////////////////////////////////////////////////
		stimgroup->clear_patterns(); sys->run(1);


		/////////////////////
		/// Testing phase ///
		///////////////////// 

		sys->save_network_state("net" + ID); // we will revert to that state for each individual stimulus

		/// PART 1: Show each stimulus in isolation:
		/////CHANGED
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		/////CHANGED
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		// PART 2: Show sequences of stimuli:
		// 2.1 Spatially and temporally familiar sequences (same as training)
		// F1-F2-F3-F4
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F2-F3-F4-F5
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F3-F4-F5-F1
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F4-F5-F1-F2
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F5-F1-F2-F3
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		// 2.2 Spatially familiar but temporally novel sequences
		// F1-F2-F3-F5
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F2-F3-F4-F1
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F3-F4-F5-F2
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F4-F5-F1-F3
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F5-F1-F2-F4
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns();sys->run(l_stim_off_test);

		// 2.3 Temporally and spatially novel sequences
		// F1-F2-F3-N1
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F2-F3-F4-N2
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F3-F4-F5-N1
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam3); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F4-F5-F1-N2
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam4); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);
		// F5-F1-F2-N1
		sys->load_network_state(workdir + "/net" + ID);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam5); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(fam2); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); stimgroup->load_patterns(nov1); stimgroup->set_mean_on_period(mean_on); stimgroup->set_mean_off_period(mean_off); sys->run(l_stim_on_test);
		stimgroup->clear_patterns(); sys->run(l_stim_off_test);

		sys->load_network_state(workdir + "/net" + ID); // revert to state before begining to continue the break time seemlessly
	}

	// Done with simulation: housekeeping and returning status
	remove( (workdir + "/net" + ID + ".0.netstate").c_str() ); // remove the network state stored, important on the cluster especially

	// Find out if rate-checker interrupted the simulation and output the info.
	float pretrain_eng_duration = 7*(l_stim_on_pretrain + l_stim_off_pretrain);
	float total_break = 0;
	for (int i=0; i<n_breaks; i++){
		total_break = total_break + break_durations[i];
	}
	float duration_singlestim_test = 7*(l_stim_on_test + l_stim_off_test);
	float duration_seqstim_test = 15*(4*l_stim_on_test + l_stim_off_test);
	float duration_1test = duration_singlestim_test + duration_seqstim_test;
	float total_sim_time = l_pre_train + pretrain_eng_duration + l_train + total_break + n_breaks*duration_1test;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// std::cout << "real simulated time: " << sys->get_time() << ", predicted time: " << total_sim_time;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sys->get_time() >= 0.99*total_sim_time){
		std::cout << "cynthia" << -1 << "cynthia";
	}
	else{
		std::cout << "cynthia" << sys->get_time() << "cynthia";
	}

	auryn_free();
	return 0;
}