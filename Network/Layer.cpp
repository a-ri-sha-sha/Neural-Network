#include "Layer.h"

namespace layer {
    Layer::Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize) : sigma_(
            std::move(sigma)), A_(GetRandomMatrix(output, input, seed, normalize)), b_(GetRandomMatrix(
                    output, 1, seed, normalize)) {
    }


    Layer::Matrix Layer::Result(const Matrix &x) const {
        return sigma_.Apply(A_ * x + b_);
    }

    Layer::Matrix Layer::GetDerA(const Matrix &x, const Matrix &u) const {
        return (u.transpose() * (x.transpose())) / x.rows();
    }

    Layer::Matrix Layer::GetDerB(const Matrix &x, const Matrix &u) const {
        return (u.transpose() * Eigen::RowVectorXd::Ones(A_.rows()).transpose());
    }

    Layer::Matrix Layer::PushU(Matrix x, Matrix u) const {
        Matrix e = Eigen::RowVectorXd::Ones(A_.rows());
        Matrix der = sigma_.Derivative(A_ * x + b_ * e);
        Matrix new_u(u.rows(), u.cols());
        new_u = u * der;
        return new_u;
    }

    void Layer::ChangeA(const Matrix &DerA, Index h) {
        A_ = A_ - h * DerA;
    }

    void Layer::ChangeB(const Matrix &DerB, Index h) {
        b_ = b_ - h * DerB;
    }

    Layer::RandGen Layer::GetUrng(int seed){
        static Layer::RandGen urng = seed;
        return urng;
    }

    Layer::Matrix Layer::GetRandomMatrix(Layer::Index rows, Layer::Index cols, int seed, float normalize) {
        return Eigen::Rand::normal<Matrix>(rows, cols, GetUrng(seed)) * normalize;
    }

}
