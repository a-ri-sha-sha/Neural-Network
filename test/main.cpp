#include "Eigen/Dense"
#include "../EigenRand/EigenRand/EigenRand"
#include "net.h"
#include "Except.h"

int main() {
    try {
//        neural_network::RunAllTests();
    } catch (...) {
        except::react();
    }
    return 0;
}
