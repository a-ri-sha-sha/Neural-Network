#include "LossFunction.h"

namespace loss_function {
    LossFunction::LossFunction(FuncDist f1, FuncDer f2) : dist_(std::move(f1)), der_(std::move(f2)) {
    }

    double LossFunction::Dist(const Vector &x, const Vector &y) const {
        return dist_(x, y);
    }

    Matrix LossFunction::Derivative(const Vector &x, const Vector &y) const {
        return der_(x, y);
    }

    double LossFunction::Dist(const Vector &x, const Matrix &y) const {
        double ret = 0;
        for (Index i = 0; i < y.cols(); ++i) {
            ret += dist_(x, y.col(i));
        }
        return ret / y.cols();
    }

    Matrix LossFunction::Derivative(const Vector &x, const Matrix &y) const {
        Matrix ret = Matrix::Zero(x.size(), x.size());
        for (Index i = 0; i < y.cols(); ++i) {
            ret += der_(x, y.col(i));
        }
        return ret / y.cols();
    }

    double MSE::Dist(const Vector &x, const Vector &y) {
        return ((x - y) * (x - y).transpose()).trace() / x.size();
    }

    Matrix loss_function::MSE::Derivative(const Vector &x, const Vector &y) {
        return 2.0 * (x - y) / x.size();
    }

    double BCELoss::Dist(const Vector &x, const Vector &y) {
        return -(y.array() * x.array().log() + (1.0 - y.array()) * (1.0 - x.array()).log()).sum() / x.size();
    }

    Matrix BCELoss::Derivative(const Vector &x, const Vector &y) {
        return (x.array() - y.array()) / (x.array() * (1.0 - x.array()) * x.size());
    }
}
