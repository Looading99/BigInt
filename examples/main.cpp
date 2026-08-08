#include <chrono>
#include <fstream>
#include <iostream>
#include <string>


#include "bigint/bigint.h"


using namespace bigint;
namespace chrono = std::chrono;


auto main() -> int {
    init_thread_pool(8);

    auto clock = chrono::high_resolution_clock();

    BigInt a(1234567);
    auto   t1 = clock.now();

    a       = fast_pow(a, 7654321);
    auto t2 = clock.now();

    auto calc_time = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    std::cout << "calc time=" << calc_time.count() << "ms  len=" << a.len() << "\n";


    std::fstream fout("output.txt", std::ios::out);

    t2      = clock.now();
    auto s  = a.to_string();
    auto t3 = clock.now();
    fout << s << '\n';
    fout.close();

    auto output_time = chrono::duration_cast<chrono::milliseconds>(t3 - t2);
    std::cout << "output time=" << output_time.count() << "ms  len=" << s.size() << "\n";
}
