#include "Layer.h"

namespace neural_network {
    Layer::Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize) : sigma_(
            std::move(sigma)), A_(GetRandomMatrix(output, input, seed, normalize)), b_(GetRandomMatrix(
                    output, 1, seed, normalize)) {
    }


    Matrix Layer::Result(const Matrix &x) const {
        return sigma_.Apply(A_ * x + b_);
    }

    Matrix Layer::MakeDerA(const Matrix &x, const Matrix &u) const {
        return (u.transpose() * (x.transpose())) / x.rows();
    }

    Matrix Layer::MakeDerB(const Matrix &x, const Matrix &u) const {
        return (u.transpose() * Eigen::RowVectorXd::Ones(A_.rows()).transpose());
    }

    Matrix Layer::PushU(const Matrix& x, const Matrix& u) const {
        return u * sigma_.Derivative((A_ * x).colwise() + b_);
    }

    void Layer::ChangeA(const Matrix &DerA, Index h) {
        A_ = A_ - h * DerA;
    }

    void Layer::ChangeB(const Matrix &DerB, Index h) {
        b_ = b_ - h * DerB;
    }

    Layer::RandGen Layer::GetUrng(int seed){
        static RandGen urng = seed;
        return urng;
    }

    Matrix Layer::GetRandomMatrix(Index rows, Index cols, int seed, float normalize) {
        return Eigen::Rand::normal<Matrix>(rows, cols, GetUrng(seed)) * normalize;
    }

}
