#include "test.h"
#include "../mnist/include/mnist/mnist_reader.hpp"

namespace mnist_test {
    void RunAllTest() {
        neural_network::ActivationFunction f1([](double x) {return 1 / (1 + exp(-x));}, [](double x){return 5*x;});
//        Network network({10, 5}, {neural_network::Sigmoid()}, 0.99);
//        Data data = LoadMnistData();
    }

    Data LoadMnistData() {
        auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");
        const auto& images = dataset.training_images;
        const auto& labels = dataset.training_labels;

        size_t num_samples = images.size();
        size_t image_size = images[0].size(); // 28x28 = 784

        Eigen::MatrixXd input(image_size, num_samples);
        Eigen::MatrixXd output(10, num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < image_size; ++j) {
                input(j, i) = images[i][j] / 255.0;
            }
        }

        output.setZero();
        for (size_t i = 0; i < num_samples; ++i) {
            output(labels[i], i) = 1.0;
        }

        return {input, output};
    }
}
