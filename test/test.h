#ifndef NEURAL_NETWORK_TEST_H
#define NEURAL_NETWORK_TEST_H

#include "../Network/net.h"

namespace mnist_test {
    using Network = neural_network::Network;
    void RunAllTest();
    Data LoadMnistData();
}



#endif //NEURAL_NETWORK_TEST_H
