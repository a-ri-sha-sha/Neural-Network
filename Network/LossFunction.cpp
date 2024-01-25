#include "LossFunction.h"

double MSE::Dist(MatrixXd x, MatrixXd y) {
    double ret = 0;
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            ret += (x(i, j) - y(i, j)) * (x(i, j) - y(i, j));
        }
    }
    return ret / x.size();
}
double BCELoss::Dist(MatrixXd x, MatrixXd y) {
    double ret = 0;
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            ret += y(i, j) * std::log(x(i, j)) + (1 - y(i, j)) * std::log(1 - x(i, j));
        }
    }
}
