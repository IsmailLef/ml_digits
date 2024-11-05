using namespace std;

#include <vector>
#include <memory>
#include "Layer.cpp"

class Model {
public:
    Model() {};
    ~Model() {};
    void addLayers(int layerLength, int nbLayers);
    int computeLayers(int nbLayers);
    void backPropagate(int expectedOutput);
    vector<Layer> layers;
};

void Model::addLayers(int layerLength, int nbLayers) {
    Layer inputLayer(TOTAL_PIXELS, 0, nullptr);
    layers.push_back(inputLayer);
    double *prevActivations = nullptr;
    double prevLayerLength = inputLayer.layerLength;
    for (int i = 0; i < nbLayers; i++) {
        Layer layer(layerLength, prevLayerLength, prevActivations);
        layers.push_back(layer);
        prevActivations = layer.activations;
        prevLayerLength = layer.layerLength;
    }
    Layer outputLayer(OUTPUT_LAYER_LENGTH, prevLayerLength, prevActivations);
    layers.push_back(outputLayer);
}

int Model::computeLayers(int nbLayers) {
    double *prevActivations = layers.at(0).activations;
    int prevLayerLength = layers.at(0).layerLength;
    for (int i = 1; i < nbLayers; i++) {
        Layer &layer = layers.at(i);
        layer.computeActivations(prevActivations);
        prevActivations = layer.activations;
        prevLayerLength = layer.layerLength;
    }

    // Find the max activation that means the output of the model
    double maxValue = 0;
    double maxIndex = 0;
    for (int i = 1; i < prevLayerLength; i++) {
        if (prevActivations[i] > maxValue) {
            maxValue = prevActivations[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

void Model::backPropagate(int expectedOutput) {
    double *prevActivations, *currentActivations;
    int nextLayerLength = OUTPUT_LAYER_LENGTH;
    for (int i = layers.size() - 1; i > 0; i--) {
        if (i == layers.size() - 1)
            layers.at(i).computeGradient(layers.at(i-1), layers.at(i));
        else
            layers.at(i).computeGradient(layers.at(i-1), layers.at(i+1), expectedOutput);
    }
}

double computeCost(double *currentActivations, int expectedOutput) {
    double expectedActivations[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expectedActivations[expectedOutput] = 1;

    double cost = 0;
    for (int i = 0; i < 10; i++) {
        cost += (expectedActivations[i] - currentActivations[i])*(expectedActivations[i] - currentActivations[i]);
    }
    return cost;
}

const int nbNeurons = 16;
const int nbLayers = 2;
const int nbBatches = DATA_LIMIT/BATCH_SIZE;
const int epochs = 1;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Missing parameters";
        return EXIT_FAILURE;
    }
    
    srand(time(0));
    string trainingData = argv[1];
    double outputLayer[OUTPUT_LAYER_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double *inputLayer[DATA_LIMIT];
    formatInput(trainingData, inputLayer);

    Model ml_model;
    ml_model.addLayers(nbNeurons, nbLayers);

    double expectedOutput, currentOutput;
    double *currentInput; 
    double cost;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (long i = 0; i < nbBatches; i++) {
            for (long j = 0; j < BATCH_SIZE; j++) {
                expectedOutput = inputLayer[i*BATCH_SIZE + j][0];
                // ml_model.layers.at(0).activations = &(inputLayer[i*BATCH_SIZE + j][1]);

                currentOutput = ml_model.computeLayers(nbLayers + 2); // +2 for the input and output layers
                ml_model.backPropagate(expectedOutput);
                
                // double cost = computeCost(ml_model.layers.back().activations, expectedOutput);
                // cout << cost << endl;

                // cout << endl << "expected " << expectedOutput << " , ";
                // cout << "current " << currentOutput << " : ";
            }

            // double cost = computeCost(ml_model.layers.back().activations, expectedOutput);
            // cout << cost << endl;

            for (int i = 1; i < nbLayers + 2; i++) {
                ml_model.layers.at(i).updateWeightsBiases(BATCH_SIZE);
            }
            double cost = computeCost(ml_model.layers.at(3).activations, expectedOutput);
            cout << cost << endl;
        }

        cout << "#########################" << endl;
    }    

    for (long i=0; i<DATA_LIMIT; i++) {
        delete[] inputLayer[i];
    }
     
    return EXIT_SUCCESS;
}