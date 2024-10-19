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
    Layer &operator=(const Layer& other);
    ~Layer();
    float *activations;
    float *biases;
    float **weights;
    int layerLength;
    void computeActivations(const float *prevActivations, int prevLayerLength);
};

// Constructor
Layer::Layer(int length, int prevLayerLength, float *prevActivations) : layerLength(length)
{
    biases = new float[length];
    weights = (float **) calloc(length, sizeof(float *));
    weights[0] = new float[prevLayerLength];
    biases[0] = (float) (rand()%10);
    for (int k = 0; k < prevLayerLength; k++)
        weights[0][k] = (float) (rand()%100) / 100.0;

    for (int i = 1; i < length; i++) {
        biases[i] = (float) (rand()%100) / 100.0;
        weights[i] = new float[prevLayerLength];
        memcpy(weights[i], weights[0], prevLayerLength*sizeof(float));
    }
    activations = new float[length];
    if (prevActivations != nullptr) {
        computeActivations(prevActivations, prevLayerLength);
    }
}

// Destructor
Layer::~Layer() {
    if (biases) 
        delete[] biases;
    if (activations)
        delete[] activations;
    for (int i = 0; i < layerLength; i++) {
        if (weights[i])
            delete[] weights[i];
    }
    if (weights) 
        free(weights);
}

// Copy constructor
Layer::Layer(Layer &layer) {
    layerLength = layer.layerLength;
    activations = new float[layerLength];
    activations = layer.activations;
    biases = new float[layerLength];
    biases = layer.biases;
    weights = (float **) calloc(layerLength, sizeof(float *));
    for (int i = 0; i < layerLength; i++) {
        weights[i] = new float[layerLength];
        memcpy(weights[i], layer.weights[i], layerLength*sizeof(float));
    }
}

// Copy assignment operator
Layer& Layer::operator=(const Layer& other) {
    if (this != &other) {
        delete[] biases;
        delete[] activations;
        for (int i = 0; i < layerLength; i++) {
            delete[] weights[i];
        }
        biases = new float[layerLength];
        activations = new float[layerLength];
        copy(other.biases, other.biases + layerLength, biases);
        copy(other.activations, other.activations + layerLength, activations);
        // Add copy statement for weights later
    }
    return *this;    
}

// Move constructor
Layer::Layer(Layer &&layer) {
    layerLength = layer.layerLength;
    activations = layer.activations;
    weights = layer.weights;
    biases = layer.biases;
}

void Layer::computeActivations(const float *prevActivations, int prevLayerLength) {
    double exp_sum = 0;
    for (int i = 0; i < layerLength; i++) {
        int activation = 0;
        for (int j = 0; j < prevLayerLength; j++) {
            activation += weights[i][j]*prevActivations[j];
        }
        activations[i] = exp(activation + biases[i]);
        exp_sum += activations[i];
    }

    // Normalize activations values through soft_max
    for (int i = 0; i < layerLength; i++) {
        activations[i] = activations[i] / exp_sum;
    }
}
