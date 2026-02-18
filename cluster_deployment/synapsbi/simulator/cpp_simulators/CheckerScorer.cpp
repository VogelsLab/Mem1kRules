/* 
* This file is part of synapsesbi, written by Basile Confavreux and Poornima Ramesh
* This file is adapted from Auryn and the RateChecker class. 
*/

#include "CheckerScorer.h"

using namespace auryn;



CheckerScorer::CheckerScorer(SpikingGroup * source, AurynFloat min, AurynFloat max, AurynFloat tau) : Checker()
{
	src = source;
	init(min,max,tau);
	time_blow_up = -1;
	state = (min + max)/2;
}

CheckerScorer::~CheckerScorer()
{
}

void CheckerScorer::init(AurynFloat min, AurynFloat max, AurynFloat tau)
{
	if ( src->evolve_locally() )
		auryn::sys->register_checker(this);
	timeconstant = tau;
	size = src->get_size();
	popmin = min;
	popmax = max;
	decay_multiplier = exp(-auryn_timestep/tau);
	reset();
}


bool CheckerScorer::propagate()
{
	state *= decay_multiplier;
	state += 1.*src->get_spikes()->size()/timeconstant/size; //updated current population firing rate
	if ( state>popmin && state<popmax ) { //network hasn't blown up
		return true; 
		}
	else  {
		time_blow_up = sys->get_time(); //network has blown up, remember when network blew up and stop simulation
		return false;

		}
}

AurynFloat CheckerScorer::get_property()
{
	return get_rate();
}

AurynFloat CheckerScorer::get_rate()
{
	return state;
}

void CheckerScorer::set_rate(AurynFloat r)
{
	state = r;
}

void CheckerScorer::reset()
{
	set_rate((popmax+popmin)/2);
}

void CheckerScorer::virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version ) 
{
	ar & state;
}

void CheckerScorer::virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version ) 
{
	ar & state;
}