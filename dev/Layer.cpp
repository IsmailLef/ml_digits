using namespace std;

#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>
#include "ingestion_module/FormatInput.hpp"

#define LEARNING_RATE 0.01

class Layer
{
public:
    Layer(int layerLength, int prevLayerLength, double *prevActivations);
    Layer(const Layer &other);
    Layer(Layer &&layer);
    Layer& operator=(Layer &layer);
    ~Layer();
    double *activations;
    double *derivActivations;
    double *biases;
    double **weights;
    double **gradientWeight;
    double *gradientBiases;
    int layerLength;
    int prevLayerLength;
    void computeActivations(const double *prevActivations);
    void computeGradient(Layer &prevLayer, Layer &nextLayer, int expectedOutput);
    double getCostDerivActivations(Layer &nextLayer, int i);
    double getCostDerivActivations_forOutputLayer(int i, int expectedOutput);
    void updateWeightsBiases(int batchSize);
};

// Constructor
Layer::Layer(int length, int prevLength, double *prevActivations) : layerLength(length), prevLayerLength(prevLength)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, std::sqrt(2.0 / prevLength));
    biases = new double[length];
    gradientBiases = (double *) calloc(length, sizeof(double ));
    weights = (double **) calloc(length, sizeof(double *));
    gradientWeight = (double **) calloc(length, sizeof(double *));

    for (int i = 0; i < length; i++) {
        biases[i] = 0.0;
        weights[i] = new double[prevLength];
        gradientWeight[i] = (double *) calloc(prevLength, sizeof(double));
        for (int j = 0; j < prevLength; j++)
            weights[i][j] = d(gen);
    }
    activations = new double[length];
    derivActivations = new double[length];
}

// Destructor
Layer::~Layer() {
    if (biases) 
        delete[] biases;
    if (gradientBiases)
        free(gradientBiases);
    if (activations)
        delete[] activations;
    if (derivActivations)
        delete[] derivActivations;
    for (int i = 0; i < layerLength; i++) {
        if (weights[i])
            delete[] weights[i];
        if (gradientWeight[i])
            free(gradientWeight[i]);
    }
    if (weights) 
        free(weights);
    if (gradientWeight)
        free(gradientWeight);
}

// Copy constructor
Layer::Layer(const Layer &other) {
    layerLength = other.layerLength;
    prevLayerLength = other.prevLayerLength;
    activations = new double[layerLength];
    derivActivations = new double[layerLength];
    biases = new double[layerLength];
    gradientBiases = (double *) calloc(layerLength, sizeof(double ));
    copy(other.activations, other.activations + layerLength, activations);
    copy(other.biases, other.biases + layerLength, biases);
    copy(other.gradientBiases, other.gradientBiases + layerLength, gradientBiases);
    weights = (double **) calloc(layerLength, sizeof(double *));
    gradientWeight = (double **) calloc(layerLength, sizeof(double *));
    for (int i = 0; i < layerLength; i++) {
        gradientWeight[i] = new double[prevLayerLength];
        weights[i] = new double[prevLayerLength];
        copy(other.gradientWeight[i], other.gradientWeight[i] + prevLayerLength, gradientWeight[i]);
        copy(other.weights[i], other.weights[i] + prevLayerLength, weights[i]);
    }
}

// Move constructor
Layer::Layer(Layer &&other) {
    layerLength = other.layerLength;
    activations = other.activations;
    derivActivations = other.derivActivations;
    weights = other.weights;
    gradientWeight = other.gradientWeight;
    biases = other.biases;
    gradientBiases = other.gradientBiases;
}

Layer &Layer::operator=(Layer &layer) {
    this->activations = layer.activations;
    this->derivActivations = layer.derivActivations;
    this->biases = layer.biases;
    this->layerLength = layer.layerLength;
    this->prevLayerLength = layer.prevLayerLength;
    this->weights = layer.weights;
    this->gradientWeight = layer.gradientWeight;
    this->gradientBiases = layer.gradientBiases;
    return *this;
}

void Layer::computeActivations(const double *prevActivations) {
    double activation;
    for (int i = 0; i < layerLength; i++) {
        activation = 0;
        for (int j = 0; j < prevLayerLength; j++) {
            activation += weights[i][j]*prevActivations[j];
        }
        // Normalize activations values using RELU function
        activation += biases[i];
        if (activation > 0) {
            activations[i] = activation;
            derivActivations[i] = 1;    
        }
        else {
            activations[i] = 0;
            derivActivations[i] = 0;
        }
    }
}

/*
    We consider in this implementation that, for layer L: 
    - dC/dw_ij[L] = activation_i[L-1]*derivActivations_j[L]*dC/da_j[L]
    Where dC/da_i[L] = sum_j(w_ij[L+1] * derivActivations_j[L+1] * dC/da_j[L+1])
    and for the output layer, dC/da_j[L] = 2(activation_j[L] - expectedOutput[j])

    (1) Therefore, in __derivCostActivations__, we'll store the part 
        derivActivations_j[L+1]*dC/da_j[L+1]
    to avoid recalculating it.
*/
void Layer::computeGradient(Layer &prevLayer, Layer &nextLayer, int expectedOutput=-1) {
    for (int i = 0; i < layerLength; i++) {
        double derivFactor = expectedOutput != -1 ? getCostDerivActivations_forOutputLayer(i, expectedOutput) : getCostDerivActivations(nextLayer, i);
        double derivActivation = derivActivations[i];
        for (int j = 0; j < prevLayer.layerLength; j++) {
            gradientWeight[i][j] += prevLayer.activations[j]*derivActivation*derivFactor; // divide by sample number to normalize
        }
        gradientBiases[i] += derivActivation*derivFactor;
        derivActivations[i] *= derivFactor; // This multiplication to avoid redundancy for upcoming layers, see (1)
    }
}

double Layer::getCostDerivActivations(Layer &nextLayer, int i) {
    double sum = 0;
    double **nextWeights = nextLayer.weights;
    double *nextDerivActivations = nextLayer.derivActivations;
    for (int j = 0; j < nextLayer.layerLength; j++) {
        sum += nextWeights[j][i]*nextDerivActivations[j];
    }
    return sum;
}

double Layer::getCostDerivActivations_forOutputLayer(int i, int expectedOutput) {
    double sum = 0;
    double expectedActivations[OUTPUT_LAYER_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expectedActivations[expectedOutput] = 1;
    for (int j = 0; j < OUTPUT_LAYER_LENGTH; j++) {
        sum += 2*(activations[j] - expectedActivations[j]);
    }
    return sum;
}


void Layer::updateWeightsBiases(int batchSize) {
    for (int i = 0; i < layerLength; i++) {
        for (int j = 0; j < prevLayerLength; j++) {
            weights[i][j] -= LEARNING_RATE*(gradientWeight[i][j] / (double) batchSize);
            // if (gradientWeight[i][j] != 0 && prevLayerLength > 700) cout << " changement : " << gradientWeight[i][j] << endl;
            gradientWeight[i][j] = 0;
        }
        biases[i] -= LEARNING_RATE*(gradientBiases[i] / (double) batchSize);
        gradientBiases[i] = 0;
    }
}