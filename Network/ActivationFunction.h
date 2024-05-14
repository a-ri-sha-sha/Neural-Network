#ifndef NEURAL_NETWORK_ACTIVATIONFUNCTION_H
#define NEURAL_NETWORK_ACTIVATIONFUNCTION_H

#include <cmath>
#include "Definisions.h"
#include <functional>

namespace neural_network {
    class ActivationFunction {
    private:
        using FunctionRtoR = std::function<double(double)>;

    public:
        ActivationFunction(FunctionRtoR f1, FunctionRtoR f2);
        Matrix Apply(const Matrix &x) const;
        Matrix Derivative(const Matrix &x) const;

    private:
        FunctionRtoR apply_;
        FunctionRtoR derivative_;
    };

    class Sigmoid : public ActivationFunction {
    public:
        Sigmoid()
                : ActivationFunction(
                [](double x) { return apply(x); },
                [](double x) { return derivative(x); }) {}

    private:
        static double apply(double x);

        static double derivative(double x);
    };

    class Tanh : public ActivationFunction {
    public:
        Tanh()
                : ActivationFunction(
                [](double x) { return apply(x); },
                [](double x) { return derivative(x); }) {}

    private:
        static double apply(double x);

        static double derivative(double x);
    };

    class ReLu : public ActivationFunction {
    public:
        ReLu()
                : ActivationFunction(
                [](double x) { return apply(x); },
                [](double x) { return derivative(x); }) {}

    private:
        static double apply(double x);

        static double derivative(double x);
    };

    class LeakyReLu : public ActivationFunction {
    public:
        LeakyReLu(double leak)
                : ActivationFunction(
                [leak](double x) { return apply(leak, x); },
                [leak](double x) { return derivative(leak, x); }) {}

    private:
        static double apply(double leak, double x);

        static double derivative(double leak, double x);
    };

    /// TODO: softmax?
}
#endif  // NEURAL_NETWORK_ACTIVATIONFUNCTION_H
