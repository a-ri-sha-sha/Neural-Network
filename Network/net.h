#ifndef NEURAL_NETWORKS_NET_H
#define NEURAL_NETWORKS_NET_H

#include <vector>
#include "ActivationFunction.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Definisions.h"

namespace neural_network {
    class Network {
    private:
        using Layer = layer::Layer;

    public:
        Network(std::vector<Index> sizes, std::vector<ActivationFunction> func,  double normalize);
        void Train(const Data &data, int epochs, double eps, Index batch_size, const LossFunction &lf,
                   int power_learning_rate = 1);
        Vector Predict(const Matrix &x);

    private:
        Vector ForwardPropagation(const Matrix &batch_input);
        void BackPropagation(const Matrix &output, const Matrix &batch_output, int epoch, int power_learning_rate,
                             const LossFunction &lf);

    private:
        std::vector<Layer> layers_;
    };
}
#endif  // NEURAL_NETWORKS_NET_H
