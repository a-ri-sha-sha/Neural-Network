#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "../eigen/Eigen/Dense"
#include "ActivationFunction.h"

using Eigen::Matrix;
using Eigen::MatrixXd;

class Layer {
public:
    Layer(ActivationFunction sigma);

    MatrixXd Result(MatrixXd x);

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

#endif  // NEURAL_NETWORK_LAYER_H
