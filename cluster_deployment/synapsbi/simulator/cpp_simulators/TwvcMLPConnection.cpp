/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*/

#include "TwvcMLPConnection.h"

using namespace auryn;

void TwvcMLPConnection::init(std::vector<float> taus,
            std::vector<std::vector<double> > _W1,
            std::vector<std::vector<double> > _W3_pre,
            std::vector<std::vector<double> > _W3_post,
            std::vector<std::vector<double> > _W4_pre,
            std::vector<std::vector<double> > _W4_post,
            double _eta,
			AurynWeight maxweight)
{
	set_max_weight(maxweight);
	set_min_weight(0.0);

	stdp_active = true;

	if ( dst->get_post_size() == 0 ) return;

	tr_pre1 = src->get_pre_trace(taus[0]);
	tr_pre2 = src->get_pre_trace(taus[1]);
	tr_post1 = dst->get_post_trace(taus[2]);
	tr_post2 = dst->get_post_trace(taus[3]);

    W1.resize(4, std::vector<double>(8, 0.0));
    W1 = _W1;

    W3_pre.resize(2, std::vector<double>(4, 0.0));
    W3_pre = _W3_pre;

    W3_post.resize(2, std::vector<double>(4, 0.0));
    W3_post = _W3_post;

    W4_pre.resize(1, std::vector<double>(2, 0.0));
    W4_pre = _W4_pre;

    W4_post.resize(1, std::vector<double>(2, 0.0));
    W4_post = _W4_post;

    x1.resize(4, 0.0);
    x3.resize(2, 0.0);

    if (_eta <= 0.0){
        stdp_active = false;
    }
    eta = _eta;
}

void TwvcMLPConnection::free()
{
}

TwvcMLPConnection::TwvcMLPConnection(SpikingGroup * source,
			NeuronGroup * destination,
			AurynWeight weight,
			AurynFloat sparseness,
			std::vector<float> taus,
			std::vector<std::vector<double> > _W1,
            std::vector<std::vector<double> > _W3_pre,
            std::vector<std::vector<double> > _W3_post,
            std::vector<std::vector<double> > _W4_pre,
            std::vector<std::vector<double> > _W4_post,
            double eta,
			AurynWeight maxweight,
			TransmitterType transmitter,
			string name) 
: DuplexConnection(source, destination, weight, sparseness, transmitter, name)
{
	init(taus, _W1, _W3_pre, _W3_post, _W4_pre, _W4_post, eta, maxweight);
}

TwvcMLPConnection::~TwvcMLPConnection()
{
	free();
}

inline AurynWeight TwvcMLPConnection::dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C)
{
    return eta*forward_pre(tr_pre1->get(pre),
                        tr_pre2->get(pre),
                        tr_post1->get(post),
                        tr_post2->get(post),
                        current_w,
                        V,
                        C);
}

inline AurynWeight TwvcMLPConnection::dw_post(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C)
{
    return eta*forward_post(tr_pre1->get(pre),
                        tr_pre2->get(pre),
                        tr_post1->get(post),
                        tr_post2->get(post),
                        current_w,
                        V,
                        C);
}

inline void TwvcMLPConnection::propagate_forward()
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

                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

inline void TwvcMLPConnection::propagate_backward()
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
                *weight += dw_post(*c, trans_post_ind, *weight, dst->mem->get(trans_post_ind),
                            dst->g_ampa->get(trans_post_ind) + dst->g_gaba->get(trans_post_ind));
 
                // clip weights if needed
                if (*weight > get_max_weight()) *weight = get_max_weight();
                if (*weight < get_min_weight()) *weight = get_min_weight();
            }
        }
    }
}

void TwvcMLPConnection::propagate()
{
	propagate_forward();
	propagate_backward();
}

double TwvcMLPConnection::forward_pre(double xpre1,
                                    double xpre2,
                                    double xpost1,
                                    double xpost2,
                                    double w, 
                                    double V, 
                                    double C) {
    //we only compute the part of the MLP that is required

    // ////////////////////////////DEBUG//////////////////
    // std::cout << "inside forward" << std::endl;
    // std::cout << "x_pre1 " << tr_pre1->get(pre)
    //         << "x_pre2 " << tr_pre2->get(pre) 
    //         << "x_post1 " << tr_post1->get(post)
    //         << "x_post2 " << tr_post2->get(post)
    //         << "w " << w
    //         << "V " << V
    //         << "C " << C << std::endl;
    // //////////////////////////////////////////////

    // First layer with shared weights
    x1[0] = sigmoid(xpre1*W1[0][0] +
            xpre2*W1[0][1] +
            xpost1*W1[0][2] +
            xpost2*W1[0][3] +
            w*W1[0][4] +
            V*W1[0][5] +
            C*W1[0][6] +
            W1[0][7]);

    x1[1] = sigmoid(xpre1*W1[1][0] +
            xpre2*W1[1][1] +
            xpost1*W1[1][2] +
            xpost2*W1[1][3] +
            w*W1[1][4] +
            V*W1[1][5] +
            C*W1[1][6] +
            W1[1][7]);

    x1[2] = sigmoid(xpre1*W1[2][0] +
            xpre2*W1[2][1] +
            xpost1*W1[2][2] +
            xpost2*W1[2][3] +
            w*W1[2][4] +
            V*W1[2][5] +
            C*W1[2][6] +
            W1[2][7]);

    x1[3] = sigmoid(xpre1*W1[3][0] +
            xpre2*W1[3][1] +
            xpost1*W1[3][2] +
            xpost2*W1[3][3] +
            w*W1[3][4] +
            V*W1[3][5] +
            C*W1[3][6] +
            W1[3][7]);

    // pre_specific part of the update graph
    x3[0] = sigmoid(W3_pre[0][0]*x1[0] +
                    W3_pre[0][1]*x1[1] +
                    W3_pre[0][2]*x1[2] +
                    W3_pre[0][3]*x1[3]);

    x3[1] = sigmoid(W3_pre[1][0]*x1[0] +
                    W3_pre[1][1]*x1[1] +
                    W3_pre[1][2]*x1[2] +
                    W3_pre[1][3]*x1[3]);
    
    return(tanh(W4_pre[0][0]*x3[0]+W4_pre[0][1]*x3[1]));
}

double TwvcMLPConnection::forward_post(double xpre1,
                                    double xpre2,
                                    double xpost1,
                                    double xpost2,
                                    double w, 
                                    double V, 
                                    double C) {
    //we only compute the part of the MLP that is required

    // First layer with shared weights
    x1[0] = sigmoid(xpre1*W1[0][0] +
            xpre2*W1[0][1] +
            xpost1*W1[0][2] +
            xpost2*W1[0][3] +
            w*W1[0][4] +
            V*W1[0][5] +
            C*W1[0][6] +
            W1[0][7]);

    x1[1] = sigmoid(xpre1*W1[1][0] +
            xpre2*W1[1][1] +
            xpost1*W1[1][2] +
            xpost2*W1[1][3] +
            w*W1[1][4] +
            V*W1[1][5] +
            C*W1[1][6] +
            W1[1][7]);

    x1[2] = sigmoid(xpre1*W1[2][0] +
            xpre2*W1[2][1] +
            xpost1*W1[2][2] +
            xpost2*W1[2][3] +
            w*W1[2][4] +
            V*W1[2][5] +
            C*W1[2][6] +
            W1[2][7]);

    x1[3] = sigmoid(xpre1*W1[3][0] +
            xpre2*W1[3][1] +
            xpost1*W1[3][2] +
            xpost2*W1[3][3] +
            w*W1[3][4] +
            V*W1[3][5] +
            C*W1[3][6] +
            W1[3][7]);


    // post_specific part of the update graph
    x3[0] = sigmoid(W3_post[0][0]*x1[0] +
                    W3_post[0][1]*x1[1] +
                    W3_post[0][2]*x1[2] +
                    W3_post[0][3]*x1[3]);

    x3[1] = sigmoid(W3_post[1][0]*x1[0] +
                    W3_post[1][1]*x1[1] +
                    W3_post[1][2]*x1[2] +
                    W3_post[1][3]*x1[3]);

    return(tanh(W4_post[0][0]*x3[0]+W4_post[0][1]*x3[1]));
    }

double TwvcMLPConnection::sigmoid(const double& z) {
    double x;
    if (z > 0.0) {
        x = 1.0 / (1.0 + std::exp(-z));
    } else {
        x = std::exp(z) / (1.0 + std::exp(z));
    }
    return x;
}