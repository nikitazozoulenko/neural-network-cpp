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
	std::cout << std::endl << std::endl << std::endl << std::endl;
	std::cout << "X =\n" << X << std::endl << std::endl;

	Z2 = X * W1;
	A2 = Z2.unaryExpr(&(NeuralNetwork::sigmoid));

	Z3 = A2 * W2;
	Y_hat = Z3.unaryExpr(&(NeuralNetwork::sigmoid));

	std::cout << "Y_hat =\n" << Y_hat << std::endl << std::endl;
}

void NeuralNetwork::backprop(MatrixBatch<SIZE_OUTPUT_LAYER> &Y)
{
	Matrix<double, BATCH_SIZE, SIZE_OUTPUT_LAYER> delta1 = (Y - Y_hat).cwiseProduct(Z3.unaryExpr(&(NeuralNetwork::sigmoid_prime)));
	Matrix<double, SIZE_HIDDEN_LAYER_1, SIZE_OUTPUT_LAYER> dJdW2 = A2.transpose() * delta1;

	auto u = (((Y - Y_hat).cwiseProduct(Z3.unaryExpr(std::ptr_fun(NeuralNetwork::sigmoid_prime))))* W2.transpose()).cwiseProduct(Z2.unaryExpr(&(NeuralNetwork::sigmoid_prime)));

	Matrix<double, SIZE_INPUT_LAYER, SIZE_HIDDEN_LAYER_1> dJdW1 = X.transpose() * u;

	std::cout << "Y =\n" << Y << std::endl << std::endl;
	std::cout << "dJdW1 =\n" << dJdW1 << std::endl << std::endl;
	std::cout << "dJdW2 =\n" << dJdW2 << std::endl << std::endl;

	W1 += scale_factor * dJdW1;
	W2 += scale_factor * dJdW2;
}

double NeuralNetwork::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoid_prime(double x)
{
	return 1.0 * exp(-x) / ((1.0 + exp(-x)) * (1.0 + exp(-x)));
}
