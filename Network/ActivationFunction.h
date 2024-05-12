#ifndef NEURAL_NETWORK_ACTIVATIONFUNCTION_H
#define NEURAL_NETWORK_ACTIVATIONFUNCTION_H

#include "Eigen/Dense"
#include <cmath>

namespace activation_function {
    using FunctionRtoR = std::function<double(double)>;
    using Matrix = Eigen::MatrixXd;

    class ActivationFunction {
    public:
        ActivationFunction(FunctionRtoR f1, FunctionRtoR f2);
        Matrix Apply(const Matrix &x) const;
        Matrix Derivative(const Matrix &x) const;

    private:
        FunctionRtoR apply_;
        FunctionRtoR derivative_;
    };

    class Sigmoid {
    public:
        static double Apply(double x);
        static double Derivative(double x);
    };

    class Tanh {
    public:
        static double Apply(double x);
        static double Derivative(double x);
    };

    class ReLU {
    public:
        static double Apply(double x);
        static double Derivative(double x);
    };

    class LeakyReLU {
    public:
        static double Apply(double x);
        static double Derivative(double x);

    private:
        constexpr static const double leaky = 0.01;
    };

    /// TODO: softmax?
}
#endif  // NEURAL_NETWORK_ACTIVATIONFUNCTION_H
