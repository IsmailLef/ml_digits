using namespace std;

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>

class Layer
{
public:
    Layer(int layerLength, int prevLayerLength, float *prevActivations);
    Layer(Layer &layer);
    Layer(Layer &&layer);
    ~Layer();
    float *activations;
    float *biases;
    float **weights;
    int layerLength;
    void computeActivations(float *prevActivations, int prevLayerLength);
};

// Constructor
Layer::Layer(int length, int prevLayerLength, float *prevActivations) : layerLength(length)
{
    weights = (float **) calloc(length, sizeof(float *));
    weights[0] = new float[prevLayerLength];
    for (int k = 0; k < length; k++)
        weights[0][k] = (float) (rand()%10);

    for (int i = 1; i < length; i++) {
        weights[i] = new float[prevLayerLength];
        memcpy(weights[i], weights[0], prevLayerLength*sizeof(float));
    }
    activations = new float[length];
    if (prevActivations != nullptr)
        computeActivations(prevActivations, prevLayerLength);
}

// Destructor
Layer::~Layer() {
    if (activations)
        free(activations);
    for (int i = 0; i < layerLength; i++) {
        if (weights[i])
            free(weights[i]);
    }
    if (weights) 
        free(weights);
}

// Copy constructor
Layer::Layer(Layer &layer) {
    layerLength = layer.layerLength;
    activations = new float[layerLength];
    activations = layer.activations;
    weights = (float **) calloc(layerLength, sizeof(float *));
    for (int i = 0; i < layerLength; i++) {
        weights[i] = new float[layerLength];
        memcpy(weights[i], layer.weights[i], layerLength*sizeof(float));
    }
}

// Move constructor
Layer::Layer(Layer &&layer) {
    layerLength = layer.layerLength;
    activations = layer.activations;
    weights = layer.weights;
}

void Layer::computeActivations(float *prevActivations, int prevLayerLength) {
    float exp_sum = 0;
    for (int i = 0; i < layerLength; i++) {
        int activation = 0;
        for (int j = 0; j < prevLayerLength; j++) {
            activation += weights[i][j]*prevActivations[j];
        }
        activations[i] = activation;
        exp_sum += exp(activation);
    }

    // Normalize activations values through soft_max
    for (int i = 0; i < layerLength; i++) {
        activations[i] = exp(activations[i]) / exp_sum;
    }
}
