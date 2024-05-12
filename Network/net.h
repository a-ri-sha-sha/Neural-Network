#ifndef NEURAL_NETWORKS_NET_H
#define NEURAL_NETWORKS_NET_H

#include <initializer_list>
#include <vector>
#include "Eigen/Dense"
#include "ActivationFunction.h"
#include "Layer.h"
#include "LossFunction.h"

namespace neural_network {
    using Matrix = Eigen::MatrixXd;
    using ActivationFunction = activation_function::ActivationFunction;
    using Layer = layer::Layer;
    using Index = Eigen::Index;
    using LossFunction = loss_function::LossFunction;

    struct Data {
        Matrix input;
        Matrix output;
    };

    class Network {
    public:
        Network(std::vector<Index> sizes, std::vector<ActivationFunction> func, int seed, double normalize);

        void Train(const Data &data, int epochs, double eps, Index batch_size, const LossFunction &lf,
                   int power_learning_rate = 1);

        Matrix Predict(const Matrix &x);

    private:
        Matrix ForwardPropagation(const Matrix &batch_input);

        void BackPropagation(const Matrix &output, const Matrix &batch_output, int epoch, int power_learning_rate,
                             const LossFunction &lf);

    private:
        std::vector<Layer> layers_;
    };

    void RunAllTests();
}
#endif  // NEURAL_NETWORKS_NET_H
