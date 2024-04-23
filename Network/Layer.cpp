#include "Layer.h"

MatrixXd layer::Layer::Result(const MatrixXd& x) {
    return sigma_.Apply(A_ * x + b_);
}

MatrixXd layer::Layer::GetDerA(MatrixXd x, MatrixXd u) {
    return (NewU(x, u).transpose() * (x.transpose())) / x.rows();
}

MatrixXd layer::Layer::GetDerB(MatrixXd x, MatrixXd u) {
    return (NewU(x, u).transpose() * Eigen::RowVectorXd::Ones(A_.rows()).transpose());
}

MatrixXd layer::Layer::NewU(MatrixXd x, MatrixXd u) {
    MatrixXd e = Eigen::RowVectorXd::Ones(A_.rows());
    MatrixXd der = sigma_.Derivative(A_ * x + b_ * e);
    MatrixXd new_u(u.rows(), u.cols());
    new_u = u * der;
    return new_u;
}

MatrixXd layer::Layer::NewA(MatrixXd x, MatrixXd u) {
    return A_ - h_ * GetDerA(x, u);
}

MatrixXd layer::Layer::NewB(MatrixXd x, MatrixXd u) {
    return b_ - h_ * GetDerB(x, u);
}

