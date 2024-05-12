#ifndef NEURAL_NETWORK_LOSSFUNCTION_H
#define NEURAL_NETWORK_LOSSFUNCTION_H

#include "Eigen/Dense"
#include "Definisions.h"

namespace neural_network {
    class LossFunction {
    private:
        using FuncDist = std::function<double(const Vector &, const Vector &)>;
        using FuncDer = std::function<Matrix(const Vector &, const Vector &)>;
    public:
        LossFunction(FuncDist f1, FuncDer f2);

        double Dist(const Vector &x, const Vector &y) const;
        Matrix Derivative(const Vector &x, const Vector &y) const;

        double Dist(const Vector &x, const Matrix &y) const;
        Matrix Derivative(const Vector &x, const Matrix &y) const;

    private:
        FuncDist dist_;
        FuncDer der_;
    };

    class MSE {
    public:
        static double Dist(const Vector &x, const Vector &y);
        static Matrix Derivative(const Vector &x, const Vector &y);
    };

    class BCELoss {
    public:
        static double Dist(const Vector &x, const Vector &y);
        static Matrix Derivative(const Vector &x, const Vector &y);
    };
}
#endif  // NEURAL_NETWORK_LOSSFUNCTION_H
