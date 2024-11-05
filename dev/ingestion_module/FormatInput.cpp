using namespace std;

#include "FormatInput.hpp"

int formatInput(string fileName, double *inputLayer[DATA_LIMIT]) {

    ifstream file(fileName);
    if (!file.is_open()) {
        cerr << "Error opening file : " << fileName;
        return EXIT_FAILURE;
    }

    string line;
    getline(file, line); // Data starts at line 2
    long dataLine = 0;
    while (getline(file, line) && dataLine < DATA_LIMIT) {
        stringstream ss(line);
        string activation;
        inputLayer[dataLine] = new double[TOTAL_PIXELS];

        getline(ss, activation, ',');
        inputLayer[dataLine][0] = stoi(activation);
        int i = 1;
        while (getline(ss, activation, ',') && i < TOTAL_PIXELS) {
            inputLayer[dataLine][i] =  (double) stoi(activation) / 255.0;
            i++;
        }
        dataLine++;
    }
    return EXIT_SUCCESS;
}
