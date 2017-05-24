#ifndef NEURAL_NETWORK_V1_H
#define NEURAL_NETWORK_V1_H

#define NUM_HIDDEN_LAYERS 1
#define NUM_LAYERS NUM_HIDDEN_LAYERS+2

#define SIZE_INPUT_LAYER 2
#define SIZE_HIDDEN_LAYER_1 3
#define SIZE_OUTPUT_LAYER 1

#define BATCH_SIZE 3;

#include <Eigen/Dense>
using Eigen::Matrix;

int variable = 0;

class NeuralNetwork
{
public:
	NeuralNetwork();
	void forward();
protected:
private:
	Matrix<double, 1, 1> X;
	Matrix<double, 1, 1> W1;
	Matrix<double, 1, 1> W2;
	Matrix<double, 1, 1> Z2;
	Matrix<double, 1, 1> Z3;
	Matrix<double, 1, 1> A2;
	Matrix<double, 1, 1> Y;
	Matrix<double, 1, 1> Y_hat;

	double sigmoid(double x);

	template<int i> void activation_function(Matrix<double, BATCH_SIZE, i> matrix);

};
#endif //NEURAL_NETWORK_V1_H