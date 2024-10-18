using namespace std;

#include <vector>
#include <memory>
#include "Layer.cpp"
#include "ingestion_module/FormatInput.hpp"

class Model {
public:
    Model() {};
    ~Model() {};
    void addLayers(int layerLength, int nbLayers);
    void computeLayers(float *inputLayer);
    vector<Layer *> layers;
};

void Model::addLayers(int layerLength, int nbLayers) {
    float *prevActivations = nullptr;
    float prevLayerLength = TOTAL_PIXELS;
    for (int i = 0; i < nbLayers; i++) {
        Layer *layer = new Layer(layerLength, prevLayerLength, prevActivations);
        layers.push_back(layer);
        prevActivations = layer->activations;
        prevLayerLength = layer->layerLength;
    }
    Layer *outputLayer = new Layer(OUTPUT_LAYER, prevLayerLength, prevActivations);
    layers.push_back(outputLayer);
}

void Model::computeLayers(float *inputLayer) {
    float *prevActivations = inputLayer;
    int prevLayerLength = TOTAL_PIXELS;
    for (Layer *layer: layers) {
        layer->computeActivations(prevActivations, prevLayerLength);
        prevActivations = layer->activations;
        prevLayerLength = layer->layerLength;
    }
}

// void Model::backPropagate(int expectedOutput, int currentOutput) {

// }

const int nbNeurons = 16;
const int nbLayers = 3;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Missing parameters";
        return EXIT_FAILURE;
    }
    
    string trainingData = argv[1];
    float outputLayer[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float *inputLayer[DATA_LIMIT];
    formatInput(trainingData, inputLayer);

    Model ml_model;
    ml_model.addLayers(nbNeurons, nbLayers);

    float expectedOutput;
    float *currentInput; 
    for (long i = 0; i < DATA_LIMIT; i++) {
        expectedOutput = inputLayer[0][0];
        currentInput = &(inputLayer[0][1]);
        // cout << "expected " << expectedOutput << endl;
        // cout << "current " << currentInput[0];
        ml_model.computeLayers(currentInput);
        for (int j = 0; j < 10; j++)
            cout << ml_model.layers.at(2)->activations[j] << endl;
    }


     
    return EXIT_SUCCESS;
}