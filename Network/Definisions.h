#ifndef NEURAL_NETWORK_DEFINISIONS_H
#define NEURAL_NETWORK_DEFINISIONS_H

#include "../eigen/Eigen/Dense"
#include <functional>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Index = Eigen::Index;

struct Data {
    Matrix input;
    Matrix output;
};

#endif //NEURAL_NETWORK_DEFINISIONS_H
