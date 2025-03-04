#include "net.h"
#include <algorithm>
#include <random>
#include <iostream>

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
        std::mt19937 g(rd());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), g);
            for (Index i = 0; i < data.input.cols(); i += batch_size) {
                Index actual_batch_size = std::min(batch_size, data.input.cols() - i);
                Matrix batch_input = data.input.block(0, i, data.input.rows(), actual_batch_size);
                Matrix batch_output = data.output.block(0, i, data.output.rows(), actual_batch_size);

                std::vector<Matrix> output(layers_.size() + 1);
                output[0] = batch_input;
                for (Index j = 0; j < layers_.size(); ++j) {
                    output[j + 1] = layers_[j].Result(output[j]);
                }

                double loss = lf.Dist(output[layers_.size()], batch_output);
                if (loss < eps) {
                    return;
                }

                Matrix u = lf.Derivative(output[layers_.size()], batch_output);
                for (int j = layers_.size() - 1; j >= 0; --j) {
//                    std::cout << sigma_.Derivative((A_ * x).colwise() + b_).rows() << ' ' << sigma_.Derivative((A_ * x).colwise() + b_).cols() <<'\n';
//                    std::cout << u.transpose().array().matrix().rows() <<' ' << u.transpose().array().matrix().cols() <<'\n';
                    Matrix DerA = layers_[j].MakeDerA(output[j], u);
                    Matrix DerB = layers_[j].MakeDerB(output[j], u);
                    double learning_rate = 1.0 / (1 + std::pow(epoch, power_learning_rate));
                    layers_[j].ChangeA(DerA, learning_rate);
                    layers_[j].ChangeB(DerB, learning_rate);
                    if (j > 0) {
                        u = layers_[j].PushU(output[j], u);
                    }
                }
            }

        }
    }

    Vector Network::Predict(const Matrix &x) {
        Vector output = x;
        for (Layer &layer: layers_) {
            output = layer.Result(output);
        }
        return output;
    }

    Matrix Network::ForwardPropagation(const Matrix &batch_input, std::vector<Matrix> &before_propagation) {
        Matrix output = batch_input;
        for (Index i = 0; i < layers_.size(); ++i) {
            before_propagation[i] = layers_[i].Result(output);
            output = before_propagation[i];
        }
        return output;
    }

    void Network::BackPropagation(const Matrix &output, const Matrix &batch_output,
                                  const std::vector<Matrix> &before_propagation,
                                  int epoch, int power_learning_rate, const LossFunction &lf) {
        Matrix u = lf.Derivative(output, batch_output);
        for (Index i = layers_.size() - 1; i >= 0; --i) {
            Matrix der_A = layers_[i].MakeDerA(before_propagation[i], u);
            Matrix der_b = layers_[i].MakeDerB(before_propagation[i], u);
            double learning_rate = 1.0 / (1 + std::pow(epoch, power_learning_rate));
            layers_[i].ChangeA(der_A, learning_rate);
            layers_[i].ChangeB(der_b, learning_rate);
            if (i > 0) {
                u = layers_[i].PushU(output, u);
            }
        }
    }
}
