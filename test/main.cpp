#include "test.h"

int main() {
    try {
        mnist_test::RunAllTest();
    } catch (...) {
//        except::react();
        return 5;
    }
    return 0;
}
