#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>


NeuralNetwork::NeuralNetwork() : W1(MatrixXd::Random(2, 3)), W2(MatrixXd::Random(3, 1))
{
	scale_factor = 1;
}

void NeuralNetwork::forward(MatrixBatch<SIZE_INPUT_LAYER> &input_matrix)
{
	X = input_matrix;

	Z2 = X * W1;
	A2 = Z2.unaryExpr(&(NeuralNetwork::sigmoid));

	Z3 = A2 * W2;
	Y_hat = Z3.unaryExpr(&(NeuralNetwork::sigmoid));
}

void NeuralNetwork::backprop(MatrixBatch<SIZE_OUTPUT_LAYER> &Y)
{
	Matrix<double, BATCH_SIZE, SIZE_OUTPUT_LAYER> m = (Y - Y_hat).cwiseProduct(Z3.unaryExpr(&(NeuralNetwork::sigmoid_prime)));
	
	dKdW2 = A2.transpose() * m;

	auto u = (((Y - Y_hat).cwiseProduct(Z3.unaryExpr(std::ptr_fun(NeuralNetwork::sigmoid_prime))))* W2.transpose()).cwiseProduct(Z2.unaryExpr(&(NeuralNetwork::sigmoid_prime)));

	dKdW1 = X.transpose() * u;


	W1 += scale_factor * dKdW1;
	W2 += scale_factor * dKdW2;
}

void NeuralNetwork::print_backprop()
{
	std::cout << "dKdW1 =\n" << dKdW1 << std::endl << std::endl;
	std::cout << "dKdW2 =\n" << dKdW2 << std::endl << std::endl;
}
void NeuralNetwork::print_forward()
{
	std::cout << "X =\n" << X << std::endl << std::endl;
	std::cout << "Y_hat =\n" << Y_hat << std::endl << std::endl;
}

double NeuralNetwork::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoid_prime(double x)
{
	return 1.0 * exp(-x) / ((1.0 + exp(-x)) * (1.0 + exp(-x)));
}
