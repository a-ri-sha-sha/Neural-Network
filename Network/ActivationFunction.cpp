#include "ActivationFunction.h"

activation_function::ActivationFunction::ActivationFunction(activation_function::FuncApply f1,
                                                            activation_function::FuncDerivative f2) : apply_(
        std::move(f1)),
                                                                                                      derivative_(
                                                                                                              std::move(
                                                                                                                      f2)) {

}

MatrixXd activation_function::ActivationFunction::Apply(MatrixXd &x) {
    x.unaryExpr(apply_);
    return x;
}

MatrixXd activation_function::ActivationFunction::Derivative(MatrixXd &x) {
    x.unaryExpr(derivative_);
    return x;
}

double activation_function::Sigmoid::Apply(double x) {
    return 1 / (1 + exp(-x));
}

double activation_function::Sigmoid::Derivative(double x) {
    double g = Apply(x);
    return g * (1 - g);
}

double activation_function::Tanh::Apply(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double activation_function::Tanh::Derivative(double x) {
    double g = Apply(x);
    return 1.0 - g * g;
}

double activation_function::ReLU::Apply(double x) {
    return x * (x > 0);
}

double activation_function::ReLU::Derivative(double x) {
    return x >= 0;
}

double activation_function::LeakyReLU::Apply(double x) {
    return std::max(leaky * x, x);
}

double activation_function::LeakyReLU::Derivative(double x) {
    return leaky * (x < 0) + (x >= 0);
}
