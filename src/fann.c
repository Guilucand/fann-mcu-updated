//Copyright (c) 2018 ETH Zurich, Ferdinand von Hagen, Michele Magno, Lukas Cavigelli

#include "fann.h"
#include "fann_structs.h"
#include "fann_net.h"
#include <arm_math.h>

fann_type temp_buffers[2][TEMP_BUFFER_SIZE];

fann_type *fann_run(const fann_network *network, fann_type *input) {

    fann_type *previous_output = input;
    fann_type *current_output = temp_buffers[0];

    for (int layer_it = 0; layer_it != network->layers_count; ++layer_it) {

        const fann_layer *layer = &network->layers[layer_it];

        fann_type *weights = layer->weights;
        int inputs_count = layer->inputs_count;

        for (int neuron_it = 0;
             neuron_it < layer->neurons_count;
             ++neuron_it, weights += inputs_count) {
            arm_dot_prod_f32(weights, previous_output, inputs_count, &current_output[neuron_it]);
            current_output[neuron_it] += layer->bias[neuron_it];
        }
        layer->activation_function(current_output, layer->neurons_count);

        previous_output = current_output;
        current_output = temp_buffers[(layer_it + 1) % 2];
    }

    /* set the output */
    return previous_output;
}
