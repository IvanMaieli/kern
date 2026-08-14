#ifndef KERN_TESTS_TEST_MACROS_HPP
#define KERN_TESTS_TEST_MACROS_HPP

#include "tests.hpp"

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(                                                       \
                stderr,                                                         \
                "CHECK failed: %s:%d: %s\n",                                    \
                __FILE__,                                                       \
                __LINE__,                                                       \
                #condition);                                                    \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

#endif // KERN_TESTS_TEST_MACROS_HPP
