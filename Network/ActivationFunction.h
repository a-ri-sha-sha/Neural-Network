#ifndef NEURAL_NETWORK_ACTIVATIONFUNCTION_H
#define NEURAL_NETWORK_ACTIVATIONFUNCTION_H

#include "../eigen/Eigen/Dense"
#include <cmath>

using Eigen::Matrix;
using Eigen::MatrixXd;

class ActivationFunction {
public:
    ActivationFunction(std::function<float(float)> f1, std::function<float(float)> f2)
        : apply_(f1), derivative_(f2) {
    }

    MatrixXd Apply(MatrixXd x);

    MatrixXd Derivative(MatrixXd x);

private:
    std::function<float(float)> apply_;
    std::function<float(float)> derivative_;
};

class Sigmoid {
public:
    double Apply(double x);

    double Derivative(double x);
};

class Tanh {
public:
    double Apply(double x);

    double Derivative(double x);
};

class ReLU {
public:
    double Apply(double x);

    double Derivative(double x);
};

class LeakyReLU {
public:
    double Apply(double x);

    double Derivative(double x);
};

/// TODO: softmax?

#endif  // NEURAL_NETWORK_ACTIVATIONFUNCTION_H
