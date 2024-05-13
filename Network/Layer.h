#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "EigenRand/EigenRand"
#include "ActivationFunction.h"
#include "Definisions.h"

namespace neural_network {
    class Layer {
    private:
        using RandGen = Eigen::Rand::Vmt19937_64;
    public:
        Layer(ActivationFunction sigma, Index input, Index output, double normalize);

        Vector Result(const Vector &x) const;
        Matrix Result(const Matrix &x) const;

        Matrix MakeDerA(const Vector &x, const RowVector &u) const;
        Vector MakeDerB(const Vector &x, const RowVector &u) const;

        Matrix MakeDerA(const Matrix &x, const Matrix &u) const;
        Vector MakeDerB(const Matrix &x, const Matrix &u) const;

        RowVector PushU(const Vector &x, const RowVector &u) const;
        Matrix PushU(const Matrix &x, const Matrix &u) const;

        void ChangeA(const Matrix &der_A, Index h);
        void ChangeB(const Vector &der_b, Index h);

    private:
        Matrix A_;
        Vector b_;
        ActivationFunction sigma_;

        RandGen &GetUrng();

        Matrix GetRandomMatrix(Index rows, Index cols, float normalize);
    };
}
#endif  // NEURAL_NETWORK_LAYER_H
