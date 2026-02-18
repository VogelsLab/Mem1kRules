/* 
* This is a part of Synapseek, written by Basile Confavreux. 
* It uses the spiking network simulator Auryn written by Friedemann Zenke.
*/

#ifndef TWVCCONNECTION_H_
#define TWVCCONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/DuplexConnection.h"
#include "auryn/Trace.h"
#include "auryn/LinearTrace.h"
#include <vector>

namespace auryn {


/*! \brief Implements a STDP rule with 2 pre traces and 2 post traces. It includes weight/voltage dependence, and a codependent plasticity term
 * Class adapted from SymmetricSTDPConnection
 *
 * tau_pre1 = a0, tau_pre2 = a1, tau_post1 = a2, tau_post2 = a3
 * 
 * dw_pre = a4 + x_post1*(a5 + a6*x_pre2) + 
 *          Vi*(a7 + x_post1*(a8 + a9*x_pre2)) +
 *          Ci*(a10 + x_post1*(a11 + a12*x_pre2)) +
 *          w*(a13 + x_post1*(a14 + a15*x_pre2)) + 
 *          w*Vi*(a16 + x_post1*(a17 + a18*x_pre2)) +
 *          w*Ci*(a19 + x_post1*(a20 + a21*x_pre2)) +
 *          + a22*w**2 + a23*w**3
 *
 * dw_post = a24 + x_pre1*(a25 + a26*x_post2) + 
 *          Ci*(a27 + x_pre1*(a28 + a29*x_post2)) +
 *          w*(a30 + x_pre1*(a31 + a32*x_post2)) +
 *          w*Ci*(a33 + x_pre1*(a34 + a35*x_post2))
 */

class TwvcConnection : public DuplexConnection
{

public:
	std::vector<AurynFloat> coeffs_pre;
	std::vector<AurynFloat> coeffs_post;

	Trace * tr_pre1;
	Trace * tr_pre2;
	Trace * tr_post1;
	Trace * tr_post2;

	inline AurynWeight dw_pre(NeuronID pre, NeuronID post, AurynWeight current_w, AurynFloat V, AurynFloat C);
	inline AurynWeight dw_post(NeuronID pre,NeuronID post, AurynWeight current_w, AurynFloat C);

	inline void propagate_forward();
	inline void propagate_backward();

	bool stdp_active;

	/*! Constructor to create a random sparse connection object and set up plasticity.
	 * @param sourceAurynWeight  the source group from where spikes are coming.
	 * @param destination the destination group where spikes are going.
	 * @param weight the initial weight of all connections.
	 * @param sparseness the connection probability for the sparse random set-up of the connections.
	 * @param coeffs parameters of the learning rule: [tau_pre1, tau_pre2, tau_post1, tau_post2, coeffs_pre[i].., coeffs_post[i]...]
	 * @param maxweight the maxium allowed weight.
	 * @param transmitter the transmitter type of the connection - by default GABA for inhibitory connection.
	 * @param name a meaningful name of the connection which will appear in debug output.
	 */
	TwvcConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness, std::vector<float> coeffs,
			AurynWeight maxweight=1.5 , TransmitterType transmitter=GABA, string name="TwvcMLPConnection");

	virtual ~TwvcConnection();
	void init(std::vector<float> coeffs, AurynWeight maxweight);
	void free();

	virtual void propagate();

};

}

#endif /*TWVCCONNECTION_H_*/