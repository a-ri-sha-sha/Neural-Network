#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "Eigen/Dense"
#include "../EigenRand/EigenRand/EigenRand"
#include "ActivationFunction.h"
#include "Definisions.h"

namespace neural_network {
    class Layer {
    private:
        using ActivationFunction = neural_network::ActivationFunction;
        using RandGen = Eigen::Rand::Vmt19937_64;
    public:
        Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize);
        Matrix Result(const Matrix& x) const;

        Matrix MakeDerA(const Matrix &x, const Matrix &u) const;
        Matrix MakeDerB(const Matrix &x, const Matrix &u) const;

        Matrix PushU(const Matrix& x, const Matrix& u) const;
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
