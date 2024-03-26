#include "Layer.h"

MatrixXd Layer::Result(MatrixXd x) {
    return sigma_.Apply(A_ * x + b_);
}

MatrixXd Layer::NewA(MatrixXd x, MatrixXd u) {
    return A_ - h_ * GetDerA(x, u);
}

MatrixXd Layer::NewB(MatrixXd x, MatrixXd u) {
    return b_ - h_ * GetDerB(x, u);
}

MatrixXd Layer::NewU(MatrixXd x, MatrixXd u) {
    MatrixXd e = Eigen::RowVectorXd::Ones(A_.rows());
    MatrixXd der = sigma_.Derivative(A_ * x + b_ * e);
    MatrixXd new_u(u.rows(), u.cols());
    for (int i = 0; i < u.rows(); ++i) {
        for (int j = 0; j < u.cols(); ++j) {
            new_u(i, j) = u(i, j) * der(i, j);
        }
    }
    return new_u;
}
MatrixXd Layer::GetDerA(MatrixXd x, MatrixXd u) {
    return (NewU(x, u).transpose() * (x.transpose())) / x.rows();
}
MatrixXd Layer::GetDerB(MatrixXd x, MatrixXd u) {
    return (NewU(x, u).transpose() * Eigen::RowVectorXd::Ones(A_.rows()).transpose())
}
