#ifndef NEURAL_NETWORKS_NET_H
#define NEURAL_NETWORKS_NET_H

#include <initializer_list>
#include "../eigen/Eigen/Dense"
#include "ActivationFunction.h"

using Eigen::Matrix;
using Eigen::MatrixXd;

class Network {
public:
    Network(
        std::initializer_list<int> layers,
        std::initializer_list<ActivationFunction> func);  // тут должны быть еще какие-то аргументы

    void train();  // тоже какие-то аргументы

private:
    //??
};

#endif  // NEURAL_NETWORKS_NET_H