/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*/

#include "TwvcConnection.h"

using namespace auryn;

void TwvcConnection::init(std::vector<float> coeffs, AurynWeight maxweight)
{
	set_max_weight(maxweight);
	set_min_weight(0.0);

	stdp_active = true;

	if ( dst->get_post_size() == 0 ) return;

	tr_pre1 = src->get_pre_trace(coeffs[0]);
	tr_pre2 = src->get_pre_trace(coeffs[1]);
	tr_post1 = dst->get_post_trace(coeffs[2]);
	tr_post2 = dst->get_post_trace(coeffs[3]);

	coeffs_pre.resize(20);
	coeffs_pre[0] = coeffs[4];
	coeffs_pre[1] = coeffs[5];
	coeffs_pre[2] = coeffs[6];
	coeffs_pre[3] = coeffs[7];
	coeffs_pre[4] = coeffs[8];
	coeffs_pre[5] = coeffs[9];
	coeffs_pre[6] = coeffs[10];
    coeffs_pre[7] = coeffs[11];
    coeffs_pre[8] = coeffs[12];
    coeffs_pre[9] = coeffs[13];
    coeffs_pre[10] = coeffs[14];
    coeffs_pre[11] = coeffs[15];
    coeffs_pre[12] = coeffs[16];
    coeffs_pre[13] = coeffs[17];
    coeffs_pre[14] = coeffs[18];
    coeffs_pre[15] = coeffs[19];
    coeffs_pre[16] = coeffs[20];
    coeffs_pre[17] = coeffs[21];
    coeffs_pre[18] = coeffs[22];
    coeffs_pre[19] = coeffs[23];

	coeffs_post.resize(18);
	coeffs_post[0] = coeffs[24];
	coeffs_post[1] = coeffs[25];
	coeffs_post[2] = coeffs[26];
	coeffs_post[3] = coeffs[27];
	coeffs_post[4] = coeffs[28];
	coeffs_post[5] = coeffs[29];
	coeffs_post[6] = coeffs[30];
    coeffs_post[7] = coeffs[31];
    coeffs_post[8] = coeffs[32];
    coeffs_post[9] = coeffs[33];
    coeffs_post[10] = coeffs[34];
    coeffs_post[11] = coeffs[35];

    // //////////////////////////////////////////////
    // std::cout << "coeff_pre" << std::endl;
    // for (int i = 0; i < 20; i ++)
    // {
    //     std::cout << coeffs_pre[i] << " ";
    // }
    // std::cout << "coeff_post" << std::endl;
    // for (int i = 0; i < 12; i ++)
    // {
    //     std::cout << coeffs_post[i] << " ";
    // }
    // //////////////////////////////////////////////
}

void TwvcConnection::free()
{
}

TwvcConnection::TwvcConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness, std::vector<float> coeffs,
			AurynWeight maxweight, TransmitterType transmitter, string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(coeffs, maxweight);
}

TwvcConnection::~TwvcConnection()
{
	free();
}

inline AurynWeight TwvcConnection::dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C)
{
	if (stdp_active) 
    {
        double dw = coeffs_pre[0] + tr_post1->get(post)*(coeffs_pre[1] + coeffs_pre[2]*tr_pre2->get(pre)) +
        V*(coeffs_pre[3] + tr_post1->get(post)*(coeffs_pre[4] + coeffs_pre[5]*tr_pre2->get(pre))) +
        C*(coeffs_pre[6] + tr_post1->get(post)*(coeffs_pre[7] + coeffs_pre[8]*tr_pre2->get(pre))) +
        current_w*(coeffs_pre[9] + tr_post1->get(post)*(coeffs_pre[10] + coeffs_pre[11]*tr_pre2->get(pre))) +
        current_w*V*(coeffs_pre[12] + tr_post1->get(post)*(coeffs_pre[13] + coeffs_pre[14]*tr_pre2->get(pre))) +
        current_w*C*(coeffs_pre[15] + tr_post1->get(post)*(coeffs_pre[16] + coeffs_pre[17]*tr_pre2->get(pre))) +
        coeffs_pre[18]*current_w*current_w + coeffs_pre[19]*current_w*current_w*current_w;
        // //////////////////////////////////////////////
        // std::cout << "inside dw_pre" << std::endl;
        // std::cout << pre << " " << post << " " << current_w << " " << V << " " << C << std::endl;
        // //////////////////////////////////////////////
        return dw;
	}
	else return 0.;
}

inline AurynWeight TwvcConnection::dw_post(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat C)
{
	if (stdp_active) 
    {
        double dw = coeffs_post[0] + tr_pre1->get(pre)*(coeffs_post[1] + coeffs_post[2]*tr_post2->get(post)) + 
        C*(coeffs_post[3] + tr_pre1->get(pre)*(coeffs_post[4] + coeffs_post[5]*tr_post2->get(post))) +
        current_w*(coeffs_post[6] + tr_pre1->get(pre)*(coeffs_post[7] + coeffs_post[8]*tr_post2->get(post))) +
        current_w*C*(coeffs_post[9] + tr_pre1->get(pre)*(coeffs_post[10] + coeffs_post[11]*tr_post2->get(post)));
		// //////////////////////////////////////////////
        // std::cout << "inside dw_post" << std::endl;
        // std::cout << pre << " " << post << " " << current_w << " " << C << std::endl;
        // //////////////////////////////////////////////
        return dw;
	}
	else return 0.;
}

inline void TwvcConnection::propagate_forward()
{
    // loop over all spikes: spike = pre_spike
    for (SpikeContainer::const_iterator spike = src->get_spikes()->begin(); spike != src->get_spikes()->end(); ++spike) 
    {
        // loop over all postsynaptic partners (c: untranslated post index)
        for (const NeuronID* c = w->get_row_begin(*spike); c != w->get_row_end(*spike); ++c) 
        {
            // transmit signal to target at postsynaptic neuron (no plasticity yet)
            AurynWeight* weight = w->get_data_ptr(c);
            transmit(*c, *weight);
 
            // handle plasticity
            if (stdp_active) 
            {
                // translate postsynaptic spike (required for mpi run)
                NeuronID trans_post_ind = dst->global2rank(*c);

                // perform weight update
                *weight += dw_pre(*spike, trans_post_ind, *weight, dst->mem->get(trans_post_ind),
                            dst->g_ampa->get(trans_post_ind) + dst->g_gaba->get(trans_post_ind));
                // //////////////////////////////////////////////
                // std::cout << "inside propagate forward" << std::endl;
                // std::cout << *spike << " " << trans_post_ind << " " << dst->mem->get(trans_post_ind) << " " << dst->g_ampa->get(trans_post_ind) << " " << dst->g_gaba->get(trans_post_ind) << std::endl;
                // //////////////////////////////////////////////

                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

inline void TwvcConnection::propagate_backward()
{
    if (stdp_active) 
    {
        SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
        
        // loop over all spikes: spike = post_spike
        for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin(); spike != spikes_end; ++spike) 
        {
            // translated id of the postsynaptic neuron that spiked
            NeuronID trans_post_ind = dst->global2rank(*spike);
 
            // loop over all presynaptic partners
            for (const NeuronID* c = bkw->get_row_begin(*spike); c != bkw->get_row_end(*spike); ++c) 
            {
                #if defined(CODE_ACTIVATE_PREFETCHING_INTRINSICS) && defined(CODE_USE_SIMD_INSTRUCTIONS_EXPLICITLY)
                // prefetches next memory cells to reduce number of last-level cache misses
                _mm_prefetch((const char *)bkw->get_data_begin()[c-bkw->get_row_begin(0)+2],  _MM_HINT_NTA);
                #endif
 
                // compute plasticity update
                AurynWeight* weight = bkw->get_data(c);
                *weight += dw_post(*c, trans_post_ind, *weight, 
                            dst->g_ampa->get(trans_post_ind) + dst->g_gaba->get(trans_post_ind));
 
                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

void TwvcConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}