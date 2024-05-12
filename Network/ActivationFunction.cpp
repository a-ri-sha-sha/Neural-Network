#include "ActivationFunction.h"

namespace activation_function {
    ActivationFunction::ActivationFunction(FunctionRtoR f1, FunctionRtoR f2) : apply_(std::move(f1)), derivative_(std::move(f2)) {

    }

    Matrix ActivationFunction::Apply(const Matrix& x) const {
        return x.unaryExpr(apply_);
    }

    Matrix ActivationFunction::Derivative(const Matrix &x) const {
        return x.unaryExpr(derivative_);
    }

    double Sigmoid::Apply(double x) {
        return 1 / (1 + exp(-x));
    }

    double Sigmoid::Derivative(double x) {
        double g = Apply(x);
        return g * (1 - g);
    }

    double Tanh::Apply(double x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double Tanh::Derivative(double x) {
        double g = Apply(x);
        return 1.0 - g * g;
    }

    double ReLU::Apply(double x) {
        return x * (x > 0);
    }

    double ReLU::Derivative(double x) {
        return x >= 0;
    }

    double LeakyReLU::Apply(double x) {
        return std::max(leaky * x, x);
    }

    double LeakyReLU::Derivative(double x) {
        return leaky * (x < 0) + (x >= 0);
    }

}
