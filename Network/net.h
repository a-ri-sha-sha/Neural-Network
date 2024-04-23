#ifndef NEURAL_NETWORKS_NET_H
#define NEURAL_NETWORKS_NET_H

#include <initializer_list>
#include "Eigen/Dense"
#include "ActivationFunction.h"

namespace neural_network {
    using MatrixXd = Eigen::MatrixXd;
    using ActivationFunction = activation_function::ActivationFunction;

    class Network {
    public:
        Network(
                std::initializer_list<int> layers,
                std::initializer_list<ActivationFunction> func);

        void train();  // тоже какие-то аргументы

    private:
        //??
    };
    void run_all_tests();
}
#endif  // NEURAL_NETWORKS_NET_H