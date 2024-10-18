using namespace std;

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

const int TOTAL_PIXELS = 784;
const int ROW_PIXELS = 28;
const int OUTPUT_LAYER = 10;
const int DATA_LIMIT = 10000;

struct dataUnit {
    int label;
    float *inputArray;
};

int formatInput(string fileName, float *inputLayer[DATA_LIMIT]);