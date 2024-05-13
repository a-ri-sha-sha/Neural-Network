#include "Layer.h"

namespace neural_network {
    Layer::Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize) : sigma_(
            std::move(sigma)), A_(GetRandomMatrix(output, input, seed, normalize)), b_(GetRandomMatrix(
            output, 1, seed, normalize)) {
    }


    Vector Layer::Result(const Vector &x) const {
        return sigma_.Apply(A_ * x + b_);
    }

    Matrix Layer::Result(const Matrix &x) const {
        return sigma_.Apply((A_ * x).colwise() + b_);
    }

    Matrix Layer::MakeDerA(const Vector &x, const RowVector &u) const {
        return sigma_.Derivative(A_ * x + b_) * u.transpose() * x.transpose();
    }

    Vector Layer::MakeDerB(const Vector &x, const RowVector &u) const {
        return sigma_.Derivative(A_ * x + b_) * u.transpose();
    }

    Matrix Layer::MakeDerA(const Matrix &x, const Matrix &u) const {
        return (sigma_.Derivative((A_ * x).colwise() + b_).array() * u.transpose().array()).matrix() * x.transpose() /
               u.rows();
    }

    Vector Layer::MakeDerB(const Matrix &x, const Matrix &u) const {
        return (sigma_.Derivative((A_ * x).colwise() + b_).array() * u.transpose().array()).rowwise().sum() / u.rows();
    }

    RowVector Layer::PushU(const Vector &x, const RowVector &u) const {
        return u * sigma_.Derivative(A_ * x + b_) * A_;
    }

    Matrix Layer::PushU(const Matrix &x, const Matrix &u) const {
        return u * sigma_.Derivative((A_ * x).colwise() + b_) * A_;
    }

    void Layer::ChangeA(const Matrix &der_A, Index h) {
        A_ = A_ - h * der_A;
    }

    void Layer::ChangeB(const Vector &der_b, Index h) {
        b_ = b_ - h * der_b;
    }

    Layer::RandGen Layer::GetUrng(int seed) {
        static RandGen urng = seed;
        return urng;
    }

    Matrix Layer::GetRandomMatrix(Index rows, Index cols, int seed, float normalize) {
        return Eigen::Rand::normal<Matrix>(rows, cols, GetUrng(seed)) * normalize;
    }

}
