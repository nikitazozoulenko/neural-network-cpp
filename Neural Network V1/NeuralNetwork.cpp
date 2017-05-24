#include "NeuralNetwork.h"
#include <cmath>

NeuralNetwork::NeuralNetwork()
{

}

void NeuralNetwork::forward(Matrix input_matrix)
{
	X = input_matrix;

}

double NeuralNetwork::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void NeuralNetwork::activation_function(Eigen::MatrixXd& matrix)
{
	return matrix.unaryExpr(&NeuralNetwork::sigmoid);
}