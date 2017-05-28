#ifndef PRODUCTINPUTCREATOR
#define PRODUCTINPUTCREATOR

#include "NeuralNetwork.h"

class ProductInputCreator
{
public:
	NeuralNetwork::MatrixBatch<2> get_input();
	NeuralNetwork::MatrixBatch<1> get_output();
	void gen_values();
protected:
private:
	NeuralNetwork::MatrixBatch<2> input;
	NeuralNetwork::MatrixBatch<1> output;
};
#endif // !PRODUCTINPUTCREATOR
