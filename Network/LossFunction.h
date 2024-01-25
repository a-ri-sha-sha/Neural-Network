#ifndef NEURAL_NETWORK_LOSSFUNCTION_H
#define NEURAL_NETWORK_LOSSFUNCTION_H

#include "../eigen/Eigen/Dense"

using Eigen::Matrix;
using Eigen::MatrixXd;

class LossFunction {
public:
    LossFunction();

    double Dist(MatrixXd x, MatrixXd y);

    MatrixXd FirstU(MatrixXd x, MatrixXd y);  // d(dist(x, y))/dx
private:
    std::function<double(MatrixXd, MatrixXd)> dist_;
    std::function<MatrixXd(MatrixXd, MatrixXd)> u_;
};

class MSE {
public:
    double Dist(MatrixXd x, MatrixXd y);

    MatrixXd FirstU(MatrixXd x, MatrixXd y);
};

class BCELoss {
public:
    double Dist(MatrixXd x, MatrixXd y);

    MatrixXd FirstU(MatrixXd x, MatrixXd y);
};

#endif  // NEURAL_NETWORK_LOSSFUNCTION_H
