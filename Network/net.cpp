#include "net.h"
#include <algorithm>
#include <random>

namespace neural_network {
    Network::Network(std::vector<Index> sizes, std::vector<ActivationFunction> func, double normalize) {
        layers_.reserve(sizes.size() - 1);
        for (Index i = 0; i < sizes.size() - 1; ++i) {
            layers_.emplace_back(func[i], sizes[i], sizes[i + 1], normalize);
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
                Matrix output = ForwardPropagation(batch_input);
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

    Matrix Network::ForwardPropagation(const Matrix &batch_input) {
        Matrix output = batch_input;
        for (Layer &layer: layers_) {
            output = layer.Result(output);
        }
        return output;
    }

    void Network::BackPropagation(const Matrix &output, const Matrix &batch_output,
                                  int epoch, int power_learning_rate, const LossFunction &lf) {
        Matrix u = lf.Derivative(output, batch_output);
        for (int i = layers_.size() - 1; i >= 0; --i) {
            Matrix der_A = layers_[i].MakeDerA(output, u);
            Matrix der_b = layers_[i].MakeDerB(output, u);
            double learning_rate = 1.0 / (1 + std::pow(epoch, power_learning_rate));
            layers_[i].ChangeA(der_A, learning_rate);
            layers_[i].ChangeB(der_b, learning_rate);
            if (i > 0) {
                u = layers_[i].PushU(output, u);
            }
        }
    }
}
