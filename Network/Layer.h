#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "Eigen/Dense"
#include "../EigenRand/EigenRand/EigenRand"
#include "ActivationFunction.h"

namespace layer {
    class Layer {
    private:
        using RandGen = Eigen::Rand::Vmt19937_64;
        using Matrix = Eigen::MatrixXd;
        using ActivationFunction = activation_function::ActivationFunction;
        using Vector = Eigen::VectorXd;
        using Index = Eigen::Index;
    public:
        Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize);

        Matrix Result(const Matrix& x) const;

        Matrix GetDerA(const Matrix &x, const Matrix &u) const;
        Matrix GetDerB(const Matrix &x, const Matrix &u) const;

        Matrix PushU(Matrix x, Matrix u) const;

        void ChangeA(const Matrix& DerA, Index h);
        void ChangeB(const Matrix& DerB, Index h);

    private:
        Matrix A_;
        Vector b_;
        ActivationFunction sigma_;
        RandGen GetUrng(int seed);
        Matrix GetRandomMatrix(Index rows, Index cols, int seed, float normalize);
    };
}
#endif  // NEURAL_NETWORK_LAYER_H
