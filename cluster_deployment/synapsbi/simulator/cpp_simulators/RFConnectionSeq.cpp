/* 
* This is adapted from STPConnection, a part of Auryn (Friedemann Zenke)
*/

#include "RFConnectionSeq.h"

using namespace auryn;

RFConnectionSeq::RFConnectionSeq(SpikingGroup * source, NeuronGroup * destination, AurynWeight weight, 
						int radius, TransmitterType transmitter, std::string name) 
						: SparseConnection(source, destination, weight, 1.0, transmitter, name)
{
	w->set_all(0.0);

	std::vector< std::vector<int> > rf_matrix;
    //rows=pre, cols=post
	rf_matrix = make_RFconnectivitySeq(get_m_rows(), get_n_cols(), radius);

	for ( NeuronID i = 0 ; i < get_m_rows() ; ++i ) 
	{
		for ( NeuronID* j = w->get_row_begin(i) ; j != w->get_row_end(i) ; ++j )
		{
			w->get_data_begin()[j-w->get_row_begin(0)] = rf_matrix[i][*j]*weight;
		}
	}
	w->prune();

    // print matrix
 //    std::cout << "rf_matrix " << get_m_rows() << " " << get_n_cols() << std::endl;
	// for ( int i = 0 ; i < 84 ; ++i )
	// {
	// 	for ( int j = 0 ; j < 84 ; ++j )
	// 	{
	// 		std::cout << rf_matrix[i*84+j][0];
	// 	}
 //        std::cout << "NEXT LINE" << std::endl;
	// }
	// std::cout << "Done with constructor" << std::endl;
}

std::vector< std::vector<int> > RFConnectionSeq::make_RFconnectivitySeq(int N_pre, int N_post, int radius)
{
	std::vector< std::vector<int> > rf_matrix(N_pre, std::vector<int>(N_post,0));
	int N_pre_2D = (int) std::sqrt(N_pre); //ADD A CHECK IF N_pre is not a square
	std::srand(std::time(0));
	int x, y;
    int x_inp_2D, y_inp_2D, inp_ind_1D;
	std::vector<int> coords(2);
    for (int post_ind = 0; post_ind < N_post; post_ind++) //for each recurrent neuron
	{
		//choose a center randomly in input space
		int cx = std::rand() % N_pre_2D;
		int cy = std::rand() % N_pre_2D;

		for (int x_lag = -radius; x_lag < radius; x_lag++)
		{
            for (int y_lag = -radius; y_lag < radius; y_lag++)
    		{
    			
    			if ((x_lag)*(x_lag) + (y_lag)*(y_lag) < radius*radius) //condition for belonging in the circle
    			{
                    // get the 2D coords of the input neuron to connect to rec_ind
                    x_inp_2D = (cx + x_lag); //can't use modulos because they can get negative in that version of C++...
                    if (x_inp_2D < 0)
                    {
                        x_inp_2D = x_inp_2D + N_pre_2D;
                    }
                    if (x_inp_2D >= N_pre_2D)
                    {
                        x_inp_2D = x_inp_2D - N_pre_2D;
                    }
                    y_inp_2D = (cy + y_lag);
                    if (y_inp_2D < 0)
                    {
                        y_inp_2D = y_inp_2D + N_pre_2D;
                    }
                    if (y_inp_2D >= N_pre_2D)
                    {
                        y_inp_2D = y_inp_2D - N_pre_2D;
                    }
                    
                    inp_ind_1D = x_inp_2D*N_pre_2D + y_inp_2D; //transform to 1D
                    // std::cout << "lags, " << x_lag << ", "<< y_lag << "current coordinates, without modulo " << cx + x_lag << ", "<< cy + y_lag << "current coordinates, " << x_inp_2D << ", "<< y_inp_2D << "current index, " << inp_ind_1D << std::endl;
    				rf_matrix[inp_ind_1D][post_ind] = 1;
    			}
            }
		}
	}
    return rf_matrix;
}