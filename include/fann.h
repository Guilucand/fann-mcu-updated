//Copyright (c) 2018 ETH Zurich, Ferdinand von Hagen, Michele Magno, Lukas Cavigelli

#ifndef FANN_FANN_H_
#define FANN_FANN_H_

#include "fann_structs.h"

/* Function: fann_run
    Will run input through the neural network, returning an array of outputs, the number of which being
    equal to the number of neurons in the output layer.

    See also:
        <fann_test>

    This function appears in FANN >= 1.0.0.
*/
// __attribute__((ramfunc))
fann_type *fann_run(const fann_network *network, fann_type * input);


#endif /* FANN_FANN_H_ */
