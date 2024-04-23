#ifndef NEURAL_NETWORK_LOSSFUNCTION_H
#define NEURAL_NETWORK_LOSSFUNCTION_H

#include "Eigen/Dense"

namespace loss_function {
    using MatrixXd = Eigen::MatrixXd;
    using FuncDist = std::function<double(const MatrixXd &, const MatrixXd &)>;
    using FuncU = std::function<MatrixXd(const MatrixXd &, const MatrixXd &)>;


    class LossFunction {
    public:
        LossFunction(FuncDist f1, FuncU f2);

        const double Dist(const MatrixXd &x, const MatrixXd &y);

        const MatrixXd FirstU(const MatrixXd &x, const MatrixXd &y);// d(dist(x, y))/dx

    private:
        FuncDist dist_;
        FuncU u_;
    };

    class MSE {
    public:
        static double Dist(const MatrixXd &x, const MatrixXd &y);

        static MatrixXd FirstU(const MatrixXd &x, const MatrixXd &y);
    };

    class BCELoss {
    public:
        static double Dist(const MatrixXd &x, const MatrixXd &y);

        static MatrixXd FirstU(const MatrixXd &x, const MatrixXd &y);
    };
}
#endif  // NEURAL_NETWORK_LOSSFUNCTION_H
