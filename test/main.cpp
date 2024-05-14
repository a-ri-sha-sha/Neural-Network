#include "test.h"
#include "../Network/Except.h"
#include <iostream>
#include <functional>


int main() {
    try {
        mnist_test::RunAllTest();
    } catch (...) {
//        except::react();
        return 5;
    }
    return 0;
}
