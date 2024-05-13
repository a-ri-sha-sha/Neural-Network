#include "net.h"
#include <algorithm>
#include <random>

namespace neural_network {
    Network::Network(std::vector<Index> sizes, std::vector<ActivationFunction> func, int seed, double normalize) {
        layers_.reserve(sizes.size() - 1);
        for (Index i = 1; i < sizes.size(); ++i) {
            layers_.emplace_back(func[i], sizes[i - 1], sizes[i], seed, normalize);
        }
    }

    void Network::Train(const Data &data, int epochs, double eps, Index batch_size, const LossFunction &lf,
                        int power_learning_rate) {
        std::vector<int> indices(data.input.cols());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), gen);
            for (Index i = 0; i < data.input.cols(); i += batch_size) {
                Index actual_batch_size = std::min(batch_size, data.input.cols() - i);
                Matrix batch_input = data.input.block(0, i, data.input.rows(), actual_batch_size);
                Matrix batch_output = data.output.block(0, i, data.output.rows(), actual_batch_size);
                Vector output = ForwardPropagation(batch_input);
                double loss = lf.Dist(output, batch_output);
                if (loss < eps) {
                    return;
                }
                BackPropagation(output, batch_output, epoch, power_learning_rate, lf);
            }
        }
    }

    Vector Network::Predict(const Matrix &x) {
        return ForwardPropagation(x);
    }

    Vector Network::ForwardPropagation(const Matrix &batch_input) {
        Matrix output = batch_input;
        for (Layer &layer: layers_) {
            output = layer.Result(output);
        }
        return output;
    }

    void Network::BackPropagation(const Vector &output, const Matrix &batch_output,
                                  int epoch, int power_learning_rate, const LossFunction &lf) {
        Matrix u = lf.Derivative(output, batch_output);
        for (int j = layers_.size() - 1; j >= 0; --j) {
            Matrix DerA = layers_[j].MakeDerA(output, u);
            Matrix DerB = layers_[j].MakeDerB(output, u);
            double learning_rate = 1.0 / (1 + std::pow(epoch, power_learning_rate));
            layers_[j].ChangeA(DerA, learning_rate);
            layers_[j].ChangeB(DerB, learning_rate);
            if (j > 0) {
                u = layers_[j].PushU(output, u);
            }
        }

    }
}
