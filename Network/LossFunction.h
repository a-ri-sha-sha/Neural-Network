#ifndef NEURAL_NETWORK_LOSSFUNCTION_H
#define NEURAL_NETWORK_LOSSFUNCTION_H

#include "Definisions.h"

namespace neural_network {
    class LossFunction {
    private:
        using FuncVectToR = std::function<double(const Vector &, const Vector &)>;
        using FuncVectToMatrix = std::function<Matrix(const Vector &, const Vector &)>;

    public:
        LossFunction(FuncVectToR f1, FuncVectToMatrix f2);

        double Dist(const Vector &x, const Vector &y) const;
        Matrix Derivative(const Vector &x, const Vector &y) const;

        double Dist(const Matrix &x, const Matrix &y) const;
        Matrix Derivative(const Matrix &x, const Matrix &y) const;

    private:
        FuncVectToR dist_;
        FuncVectToMatrix der_;
    };

    class MSE : public LossFunction {
    public:
        MSE() : LossFunction(
                [](const Vector &x, const Vector &y) { return Dist(x, y); },
                [](const Vector &x, const Vector &y) { return Derivative(x, y); }) {}

    private:
        static double Dist(const Vector &x, const Vector &y);
        static Matrix Derivative(const Vector &x, const Vector &y);
    };

    class BCELoss : public LossFunction {
    public:
        BCELoss() : LossFunction(
                [](const Vector &x, const Vector &y) { return Dist(x, y); },
                [](const Vector &x, const Vector &y) { return Derivative(x, y); }) {}

    private:
        static double Dist(const Vector &x, const Vector &y);
        static Matrix Derivative(const Vector &x, const Vector &y);
    };
}
#endif  // NEURAL_NETWORK_LOSSFUNCTION_H
