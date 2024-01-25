#include "ActivationFunction.h"

MatrixXd ActivationFunction::Apply(MatrixXd x) {
    //    MatrixXd ret(x.rows(), x.cols());
    //    ret.unaryExpr(apply_(x)); /// ?!?!??!?!
    //    return ret;
    x.unaryExpr(apply_);
    return x;
}
MatrixXd ActivationFunction::Derivative(MatrixXd x) {
    x.unaryExpr(derivate_);
    return x;
}
double Sigmoid::Apply(double x) {
    return 1 / (1 + exp(-x));
}
double Sigmoid::Derivative(double x) {
    return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
}
double Tanh::Apply(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
double Tanh::Derivative(double x) {
    return 4 * exp(2 * x) / (exp(2 * x) + 1);
}
double ReLU::Apply(double x) {
    if (x < 0) {
        return 0;
    }
    return x;
}
double ReLU::Derivative(double x) {
    if (x < 0) {
        return 0;
    }
    return 1;
}
double LeakyReLU::Apply(double x) {
    return std::max(0.01 * x, x);
}

double LeakyReLU::Derivative(double x) {
    if (x < 0) {
        return 0.01;
    }
    return 1;
}
