/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*/

#ifndef TWVCMLPCONNECTION_H_
#define TWVCMLPCONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/DuplexConnection.h"
#include "auryn/Trace.h"
#include "auryn/LinearTrace.h"
#include <vector>

namespace auryn {

/*! \brief Implements a STDP plasticity rule parameterized as an mutlit layer perceptron (class adapted from SymmetricSTDPConnection)
 * Inputs to the rule (7 + 1 bias):
 * 2 pre traces and 2 post traces, current weight, post voltage, and a codependent plasticity term (post neurons absolute sum of currents)
 *
 * tau_pre1 = a0, tau_pre2 = a1, tau_post1 = a2, tau_post2 = a3
 * 
 * Shared weights across all rules and pre/post updates: W1
 * Rule-specific weights: W3_pre/W3_post W4_pre/W4_post
 *                                 7 inputs + 1 bias
 *                                         ||
 *                                fully connected (W1)
                                        sigmoid
 *                                         ||
 *                                         \/
 *                                   4 hidden neurons
 *                                         ||
 *                            "masking layer"\"context switching"
 * makes the irrelevant part of computational graph zero (pre part on post update and vice versa)
 *                                         ||
 *                    --------------------------------------------
 *                    ||                                        ||
 *                    \/                                        \/
 *             4 hidden neurons                           4 hidden neurons
 *                    ||                                        ||
 *           fully connected (W3_pre)                  fully connected (W3_post)
 *                  sigmoid                                   sigmoid
 *                    ||                                        ||
 *                    \/                                        \/
 *               2 hidden neurons                         2 hidden neurons
 *                    ||                                        ||
 *                    \/                                        \/
 *            fully connected (W4_pre)                 fully connected (W4_post)
 *                   tanh                                      tanh
 *                    ||                                        ||
 *                    \/                                        \/
 *             1 output neuron (dw_pre)               1 output neuron (dw_post)
 *                
 *                                   weight update: eta*dw
 */

class TwvcMLPConnection : public DuplexConnection
{

public:
	//weight matrices
	std::vector<std::vector<double> > W1;
	std::vector<std::vector<double> > W3_pre;
	std::vector<std::vector<double> > W3_post;
	std::vector<std::vector<double> > W4_pre;
	std::vector<std::vector<double> > W4_post;
	
	//neurons activities
	std::vector<double> x1;
	std::vector<double> x3;
	
	//pre/post-synaptic traces
	Trace * tr_pre1;
	Trace * tr_pre2;
	Trace * tr_post1;
	Trace * tr_post2;

	inline AurynWeight dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C);
	inline AurynWeight dw_post(NeuronID pre,NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C);

	inline void propagate_forward();
	inline void propagate_backward();

	bool stdp_active;
	double eta;

	/*! Constructor to create a random sparse connection object and set up plasticity.
	 * @param source the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param weight the initial weight of all connections (of the neuronal network).
	 * @param sparseness the connection probability for the sparse random set-up of the connections (neuronal network).
	 * @param taus parameters of the timeconstant of the learning rule: [tau_pre1, tau_pre2, tau_post1, tau_post2, coeffs_pre[i].., coeffs_post[i]...]
	 * @param _W1 parameters of the learning rule: shared weights
	 * @param _W3_pre parameters of the learning rule
	 * @param _W3_post parameters of the learning rule
	 * @param _W4_pre parameters of the learning rule
	 * @param _W4_post parameters of the learning rule
	 * @param eta learning rate in front of MLP
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	TwvcMLPConnection(SpikingGroup * source,
			NeuronGroup * destination,
			AurynWeight weight,
			AurynFloat sparseness,
			std::vector<float> taus,
			std::vector<std::vector<double> > _W1,
            std::vector<std::vector<double> > _W3_pre,
			std::vector<std::vector<double> > _W3_post,
            std::vector<std::vector<double> > _W4_pre,
			std::vector<std::vector<double> > _W4_post,
			double _eta,
			AurynWeight maxweight=1.5,
			TransmitterType transmitter=GABA,
			string name="TwvcMLPConnection");

	virtual ~TwvcMLPConnection();

	void init(std::vector<float> taus,
            std::vector<std::vector<double> > _W1,
            std::vector<std::vector<double> > _W3_pre,
			std::vector<std::vector<double> > _W3_post,
            std::vector<std::vector<double> > _W4_pre,
			std::vector<std::vector<double> > _W4_post,
			double _eta,
			AurynWeight maxweight);

	void free();

	virtual void propagate();

	double forward_pre(double xpre1,
					double xpre2,
					double xpost1,
					double xpost2,
                    double w, 
                    double V, 
                    double C);

	double forward_post(double xpre1,
					double xpre2,
					double xpost1,
					double xpost2,
                    double w, 
                    double V, 
                    double C);

	double sigmoid(const double& x);
};

}

#endif /*TWVCMLPCONNECTION_H_*/