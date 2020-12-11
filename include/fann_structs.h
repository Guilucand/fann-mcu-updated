//Copyright (c) 2018 ETH Zurich, Ferdinand von Hagen, Michele Magno, Lukas Cavigelli
#include <math.h>
#include "fann.h"

#ifndef FANN_FANN_STRUCTS_H_
#define FANN_FANN_STRUCTS_H_

#define fann_max(a, b) ((a) > (b) ? (a) : (b))

typedef void (*fann_activation_function)(fann_type *, int);

typedef struct {
    int inputs_count;
    int neurons_count;
    fann_type *weights;
    fann_type *bias;
    fann_activation_function activation_function;
} fann_layer;


static void fann_activation_softmax(fann_type *data, int size) {

    fann_type max = data[0];
    for (int i = 1; i < size; i++) {
        max = fann_max(max, data[i]);
    }

    fann_type sum = 0;

    for (int i = 0; i < size; i++) {
        data[i] = expf(data[i] - max);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

static void fann_activation_tanh(fann_type *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = tanhf(data[i]);
    }
}



//#include <math.h>
//
///* FANN_LINEAR */
///* #define fann_linear(steepness, sum) fann_mult(steepness, sum) */
//
///* FANN_SIGMOID */
///* #define fann_sigmoid(steepness, sum) (1.0f/(1.0f + exp(-2.0f * steepness * sum))) */
//#define fann_sigmoid_real(sum) (1.0f/(1.0f + exp(-2.0f * sum)))
//#define fann_sigmoid_derive(steepness, value) (2.0f * steepness * value * (1.0f - value))
//
///* FANN_SIGMOID_SYMMETRIC */
///* #define fann_sigmoid_symmetric(steepness, sum) (2.0f/(1.0f + exp(-2.0f * steepness * sum)) - 1.0f) */
//#define fann_sigmoid_symmetric_real(sum) (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f)
//#define fann_sigmoid_symmetric_derive(steepness, value) steepness * (1.0f - (value*value))
//
///* FANN_GAUSSIAN */
///* #define fann_gaussian(steepness, sum) (exp(-sum * steepness * sum * steepness)) */
//#define fann_gaussian_real(sum) (exp(-sum * sum))
//#define fann_gaussian_derive(steepness, value, sum) (-2.0f * sum * value * steepness * steepness)
//
///* FANN_GAUSSIAN_SYMMETRIC */
///* #define fann_gaussian_symmetric(steepness, sum) ((exp(-sum * steepness * sum * steepness)*2.0)-1.0) */
//#define fann_gaussian_symmetric_real(sum) ((exp(-sum * sum)*2.0f)-1.0f)
//#define fann_gaussian_symmetric_derive(steepness, value, sum) (-2.0f * sum * (value+1.0f) * steepness * steepness)
//
///* FANN_ELLIOT */
///* #define fann_elliot(steepness, sum) (((sum * steepness) / 2.0f) / (1.0f + fann_abs(sum * steepness)) + 0.5f) */
//#define fann_elliot_real(sum) (((sum) / 2.0f) / (1.0f + fann_abs(sum)) + 0.5f)
//#define fann_elliot_derive(steepness, value, sum) (steepness * 1.0f / (2.0f * (1.0f + fann_abs(sum)) * (1.0f + fann_abs(sum))))
//
///* FANN_ELLIOT_SYMMETRIC */
///* #define fann_elliot_symmetric(steepness, sum) ((sum * steepness) / (1.0f + fann_abs(sum * steepness)))*/
//#define fann_elliot_symmetric_real(sum) ((sum) / (1.0f + fann_abs(sum)))
//#define fann_elliot_symmetric_derive(steepness, value, sum) (steepness * 1.0f / ((1.0f + fann_abs(sum)) * (1.0f + fann_abs(sum))))
//
///* FANN_SIN_SYMMETRIC */
//#define fann_sin_symmetric_real(sum) (sin(sum))
//#define fann_sin_symmetric_derive(steepness, sum) (steepness*cos(steepness*sum))
//
///* FANN_COS_SYMMETRIC */
//#define fann_cos_symmetric_real(sum) (cos(sum))
//#define fann_cos_symmetric_derive(steepness, sum) (steepness*-sin(steepness*sum))
//
///* FANN_SIN */
//#define fann_sin_real(sum) (sin(sum)/2.0f+0.5f)
//#define fann_sin_derive(steepness, sum) (steepness*cos(steepness*sum)/2.0f)
//
///* FANN_COS */
//#define fann_cos_real(sum) (cos(sum)/2.0f+0.5f)
//#define fann_cos_derive(steepness, sum) (steepness*-sin(steepness*sum)/2.0f)
//
///* FANN_TANH */
//#define fann_tanh_real(sum) (tanhf(sum))
//#define fann_tanh_derive(steepness, sum) (steepness*(1.0f - tanh(sum)*tanhf(sum)))

#endif /* FANN_FANN_STRUCTS_H_ */
