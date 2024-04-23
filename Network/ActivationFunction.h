#ifndef NEURAL_NETWORK_ACTIVATIONFUNCTION_H
#define NEURAL_NETWORK_ACTIVATIONFUNCTION_H

#include "Eigen/Dense"
#include <cmath>

using MatrixXd = Eigen::MatrixXd;

namespace activation_function {

    using FuncApply = std::function<double(double)>;
    using FuncDerivative = std::function<double(double)>;

    class ActivationFunction {
    public:
        ActivationFunction(FuncApply f1, FuncDerivative f2);

        MatrixXd Apply(MatrixXd &x);

        MatrixXd Derivative(MatrixXd &x);

    private:
        std::function<double(double)> apply_;
        std::function<double(double)> derivative_;
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
