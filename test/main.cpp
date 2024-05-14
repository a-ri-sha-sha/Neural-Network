#include "test.h"
#include "../Network/Except.h"

int main() {
    try {
        mnist_test::RunAllTest();
    } catch (...) {
        except::react();
    }
    return 0;
}
