#include "LossFunction.h"

loss_function::LossFunction::LossFunction(loss_function::FuncDist f1, loss_function::FuncU f2) : dist_(std::move(f1)),
                                                                                                 u_(std::move(f2)) {
}

const double loss_function::LossFunction::Dist(const loss_function::MatrixXd &x, const loss_function::MatrixXd &y) {
    return dist_(x, y);
}

const loss_function::MatrixXd
loss_function::LossFunction::FirstU(const loss_function::MatrixXd &x, const loss_function::MatrixXd &y) {
    return u_(x, y);
}

double loss_function::MSE::Dist(const MatrixXd &x, const MatrixXd &y) {
    return ((x - y) * (x - y).transpose()).trace() / x.size();
}

loss_function::MatrixXd loss_function::MSE::FirstU(const loss_function::MatrixXd &x, const loss_function::MatrixXd &y) {
    return 2.0 * (x - y) / x.size();
}

double loss_function::BCELoss::Dist(const MatrixXd &x, const MatrixXd &y) {
    return -(y.array() * x.array().log() + (1.0 - y.array()) * (1.0 - x.array()).log()).sum() / x.size();
}

loss_function::MatrixXd
loss_function::BCELoss::FirstU(const loss_function::MatrixXd &x, const loss_function::MatrixXd &y) {
    return (x.array() - y.array()) / (x.array() * (1.0 - x.array()) * x.size()) ;
}


