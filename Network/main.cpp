#include "Eigen/Dense"
#include "EigenRand/EigenRand"
#include "net.h"
#include "Except.h"

int main() {
    try {
        neural_network::run_all_tests();
    } catch (...) {
        except::react();
    }
    return 0;
}