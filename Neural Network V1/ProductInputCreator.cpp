#include "ProductInputCreator.h"

NeuralNetwork::MatrixBatch<2> ProductInputCreator::get_input()
{
	return input;
}

NeuralNetwork::MatrixBatch<1> ProductInputCreator::get_output()
{
	return output;
}

void ProductInputCreator::gen_values()
{
	input = NeuralNetwork::MatrixBatch<2>::Random(NeuralNetwork::BATCH_SIZE, 2);
	output = input.col(0).cwiseProduct(input.col(1));
}