using namespace std;

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

const int TOTAL_PIXELS = 784;
const int ROW_PIXELS = 28;
const int OUTPUT_LAYER_LENGTH = 10;
const int DATA_LIMIT = 40000;
const int BATCH_SIZE = 1000;


int formatInput(string fileName, double *inputLayer[DATA_LIMIT]);