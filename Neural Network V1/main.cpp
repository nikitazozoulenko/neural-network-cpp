#include <iostream>
#include <Eigen/Dense>

#include "NeuralNetwork.h"
#include <string>

#include "ProductInputCreator.h"

using Eigen::Matrix;
using Eigen::MatrixXd;

template <size_t N>
using MatrixBatch = NeuralNetwork::MatrixBatch<N>;
template <size_t N>
using MatrixSingle = NeuralNetwork::MatrixSingle<N>;

int main()
{
	NeuralNetwork network;

	ProductInputCreator creator;

	creator.gen_values();
	MatrixBatch<2> input = creator.get_input();
	MatrixBatch<1> output = creator.get_output();
	network.forward(input);
	network.backprop(output);
	network.print_forward();
	network.print_backprop();

	std::cout << std::endl << std::endl << std::endl << std::endl;
	for (int i = 0; i != 10000; ++i)
	{
		creator.gen_values();
		MatrixBatch<2> input = creator.get_input();
		MatrixBatch<1> output = creator.get_output();
		network.forward(input);
		network.backprop(output);
		
	}

	MatrixBatch<2> i;
	i << 1.0, 1.0,
		 1.0, 0.5,
		 0.5, 0.5;

	MatrixBatch<1> o;
	o << 1.0,
		 0.5,
		 0.25;

	network.forward(i);
	network.backprop(o);
	network.print_forward();
	network.print_backprop();

	std::cin.get();

	return 0;
}


void print(std::string name, MatrixXd m)
{
	std::cout << name + " =\n" << m << std::endl << std::endl;
}





//using Eigen::MatrixXd;
//
//double sigmoid(double x)
//{
//	return 1.0 / (1.0 + exp(-x));
//}
//
//void f(Eigen::MatrixXd& matrix)
//{
//	matrix.unaryExpr(&sigmoid);
//}
//
//
//int main()
//{
//	MatrixXd m(2, 2);
//	m(0, 0) = 3;
//	m(1, 0) = 2.5;
//	m(0, 1) = -1;
//	m(1, 1) = m(1, 0) + m(0, 1);
//
//	MatrixXd m2(2, 2);
//	m(0, 0) = 3;
//	m(1, 0) = 2.5;
//	m(0, 1) = -1;
//	m(1, 1) = m(1, 0) + m(0, 1);
//	std::cout << m << std::endl;
//
//
//		std::cout << m << std::endl;
//
//	m=m.unaryExpr(&sigmoid);
//	std::cout << m << std::endl;
//	std::cin.get();
//}


