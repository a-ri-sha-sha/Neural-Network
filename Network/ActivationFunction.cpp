#include "ActivationFunction.h"

namespace neural_network {
    ActivationFunction::ActivationFunction(FunctionRtoR f1, FunctionRtoR f2) : apply_(std::move(f1)), derivative_(std::move(f2)) {

    }

    Matrix ActivationFunction::Apply(const Matrix& x) const {
        return x.unaryExpr(apply_);
    }

    Matrix ActivationFunction::Derivative(const Matrix &x) const {
        return x.unaryExpr(derivative_);
    }

    double Sigmoid::apply(double x) {
        return 1 / (1 + exp(-x));
    }

    double Sigmoid::derivative(double x) {
        double g = apply(x);
        return g * (1 - g);
    }

    double Tanh::apply(double x) {
        return tanh(x);
    }

    double Tanh::derivative(double x) {
        return 1.0 - tanh(x) * tanh(x);
    }

    double ReLu::apply(double x) {
        return x * (x > 0);
    }

    double ReLu::derivative(double x) {
        return x >= 0;
    }

    double LeakyReLu::apply(double leak, double x) {
        return std::max(leak * x, x);
    }

    double LeakyReLu::derivative(double leak, double x) {
        return leak * (x < 0) + (x >= 0);
    }
}
