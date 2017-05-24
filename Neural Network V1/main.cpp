#include <iostream>
#include <Eigen/Dense>

#include "NeuralNetwork.h"



//int main()
//{
//	NeuralNetwork network();
//
//	return 0;
//}
using Eigen::MatrixXd;

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void f(Eigen::MatrixXd& matrix)
{
	matrix.unaryExpr(&sigmoid);
}


int main()
{
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;

	m=m.unaryExpr(&sigmoid);
	std::cout << m << std::endl;
	std::cin.get();
}


