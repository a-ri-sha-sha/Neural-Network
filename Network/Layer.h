#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "Eigen/Dense"
#include "../EigenRand/EigenRand/EigenRand"
#include "ActivationFunction.h"
#include "Definisions.h"

namespace neural_network {
    class Layer {
    private:
        using RandGen = Eigen::Rand::Vmt19937_64;
    public:
        Layer(ActivationFunction sigma, Index input, Index output, int seed, double normalize);

        Vector Result(const Vector& x) const;
        Matrix Result(const Matrix& x) const;

        Matrix MakeDerA(const Vector &x, const RowVector &u) const;
        Vector MakeDerB(const Vector &x, const RowVector &u) const;

        Matrix MakeDerA(const Matrix &x, const Matrix &u) const;
        Vector MakeDerB(const Matrix &x, const Matrix &u) const;

        RowVector PushU(const Vector & x, const RowVector & u) const;
        Matrix PushU(const Matrix& x, const Matrix& u) const;

        void ChangeA(const Matrix& der_A, Index h);
        void ChangeB(const Vector& der_b, Index h);

    private:
        Matrix A_;
        Vector b_;
        ActivationFunction sigma_;
        RandGen GetUrng(int seed);
        Matrix GetRandomMatrix(Index rows, Index cols, int seed, float normalize);
    };
}
#endif  // NEURAL_NETWORK_LAYER_H
