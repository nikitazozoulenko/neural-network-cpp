#ifndef NEURAL_NETWORK_V1_H
#define NEURAL_NETWORK_V1_H

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;

class NeuralNetwork
{
public:
	enum {
		NUM_HIDDEN_LAYERS = 1,
		NUM_LAYERS = NUM_HIDDEN_LAYERS + 2,

		SIZE_INPUT_LAYER = 2,
		SIZE_HIDDEN_LAYER_1 = 3,
		SIZE_OUTPUT_LAYER = 1,

		BATCH_SIZE = 3
	};

	template <size_t N>
	using MatrixBatch = Eigen::Matrix<double, BATCH_SIZE, N>;
	template <size_t N>
	using MatrixSingle = Eigen::Matrix<double, 1, N>;

	NeuralNetwork();

	void forward(MatrixBatch<SIZE_INPUT_LAYER> &input_matrix);
	void backprop(MatrixBatch<SIZE_OUTPUT_LAYER> &Y);

	MatrixBatch<SIZE_OUTPUT_LAYER> get_Y_hat() { return Y_hat; }

protected:
private:
	static double sigmoid(double x);
	static double sigmoid_prime(double x);

	double scale_factor;

	Matrix<double, SIZE_INPUT_LAYER, SIZE_HIDDEN_LAYER_1> W1;
	Matrix<double, SIZE_HIDDEN_LAYER_1, SIZE_OUTPUT_LAYER> W2;


	MatrixBatch<SIZE_INPUT_LAYER> X;
	MatrixBatch<SIZE_HIDDEN_LAYER_1> Z2;
	MatrixBatch<SIZE_HIDDEN_LAYER_1> A2;
	MatrixBatch<SIZE_OUTPUT_LAYER> Z3;
	MatrixBatch<SIZE_OUTPUT_LAYER> Y_hat;
};

//template <size_t N>
//using MatrixBatch = Eigen::Matrix<double, BATCH_SIZE, N>;



#endif //NEURAL_NETWORK_V1_H
