#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "Eigen/Dense"
#include "ActivationFunction.h"

namespace layer {
    using MatrixXd = Eigen::MatrixXd;
    using ActivationFunction =  activation_function::ActivationFunction;

    class Layer {
    public:
        Layer(ActivationFunction sigma, size_t cols, size_t rows);

        MatrixXd Result(const MatrixXd& x);

        MatrixXd GetDerA(MatrixXd x, MatrixXd u);

        MatrixXd GetDerB(MatrixXd x, MatrixXd u);

        MatrixXd NewU(MatrixXd x, MatrixXd u);

        MatrixXd NewA(MatrixXd x, MatrixXd u);

        MatrixXd NewB(MatrixXd x, MatrixXd u);

    private:
        MatrixXd A_;
        MatrixXd b_;
        ActivationFunction sigma_;
        double h_;  // должно же быть тут, да?
    };
}
#endif  // NEURAL_NETWORK_LAYER_H
