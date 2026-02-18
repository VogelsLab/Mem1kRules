/* 
* This is adapted from STPConnection, a part of Auryn (Friedemann Zenke)
*/

#ifndef RFCONNECTIONSEQ_H_
#define RFCONNECTIONSEQ_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/SparseConnection.h"

namespace auryn {

/*! \brief This class implements a receptive field connectivity
 * 
 * Each post neuron will be connected to a random neuron in the input layer, and all neurons in a given radius and fixed weight. 
 * (input layer considered to be sqrt(N)*sqrt(N) 2D layer
 *
 */

class RFConnectionSeq : public SparseConnection
{

public:

	/*! Default constructor to initialize connection with random recptive field connectivity
	 * \param source The presynaptic SpikingGroup
	 * \param destination the postsynaptic NeuronGroup
	 * \param weight The default weight for connections 
	 * \param radius radius of the receptive field
	 * \param transmitter The transmitter type
	 * \param name The connection name as it appears in debugging output
	 */
	RFConnectionSeq(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, int radius=8, TransmitterType transmitter=GLUT, string name="RFConnection");

    /*! returns 2D vector which will have binary weights with RFconnectivity, with donut topology compared to above
	 * \param N_pre number of pre neurons
	 * \param N_post number of post neurons 
	 * \param radius radius of the receptive field (unit = number of neurons)
	 */
	std::vector< std::vector<int> > make_RFconnectivitySeq(int N_pre, int N_post, int radius);
};

}

#endif /*RFCONNECTIONSEQ_H_*/
