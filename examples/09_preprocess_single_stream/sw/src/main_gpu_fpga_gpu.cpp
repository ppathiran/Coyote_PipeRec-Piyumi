/**
  * Copyright (c) 2021-2024, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

#include <any>
#include <iostream>
#include <cstdlib>

// AMD GPU management & run-time libraries
#include <hip/hip_runtime.h>

// External library for easier parsing of CLI arguments by the executable
#include <boost/program_options.hpp>

// Coyote-specific includes
#include "cBench.hpp"
#include "cThread.hpp"

// Constants
#define N_LATENCY_REPS 1
#define N_THROUGHPUT_REPS 1 //64

#define DEFAULT_GPU_ID 0
#define DEFAULT_VFPGA_ID 0

double run_bench(
    std::unique_ptr<coyote::cThread<std::any>> &coyote_thread, coyote::sgEntry &sg, 
    int *src_mem, int *dst_mem, uint transfers, uint n_runs
) {
    // Randomly set the source data for functional verification
    assert(sg.local.src_len == sg.local.dst_len);
    
    /*
    for (int i = 0; i < sg.local.src_len / sizeof(int); i++) {
        src_mem[i] = rand() % 1024 - 512;     
        dst_mem[i] = 0;                        
    }
    */

    
    int batch_size = (sg.local.src_len / sizeof(int)) / 48;
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
            int base_idx = sample_idx * 48;

            // --- Packet 1: 1 label + 13 dense + 2 padding ---
            src_mem[base_idx + 0] = rand() % 2;

            for (int i = 0; i < 13; ++i) {
                src_mem[base_idx + 1 + i] = rand() % 512 - 256;
            }

            src_mem[base_idx + 14] = 0;
            src_mem[base_idx + 15] = 0;

            // --- Packet 2: 16 sparse ---
            for (int i = 0; i < 16; ++i) {
                src_mem[base_idx + 16 + i] = rand() % 100000;
            }

            // --- Packet 3: 10 sparse + 6 padding ---
            for (int i = 0; i < 10; ++i) {
                src_mem[base_idx + 32 + i] = rand() % 100000;
            }

            for (int i = 0; i < 6; ++i) {
                src_mem[base_idx + 42 + i] = 0;
            }

            // Optional: clear destination memory
            for (int i = 0; i < 48; ++i) {
                dst_mem[base_idx + i] = 0;

            }
    }
    

    std::cout << "\n[C++] Data generated (first 48 values):" << std::endl;
        for (int i = 0; i < 48; ++i) {
            std::cout << src_mem[i] << " ";
        }
    std::cout << std::endl;

    auto prep_fn = [&]() {
        // Clear any previous completion flags
        coyote_thread->clearCompleted();
    };

    // Execute benchmark
    coyote::cBench bench(n_runs);
    auto bench_fn = [&]() {
        // Launch (queue) multiple transfers in parallel for throughput tests, or 1 in case of latency tests
        // Recall, coyote_thread->invoke is asynchronous (but can be made sync through different sgFlags)
        for (int i = 0; i < transfers; i++) {
            coyote_thread->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg);
        }

        // Wait until all of them are finished
        while (coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) != transfers) {}
    };
    bench.execute(bench_fn, prep_fn);

    std::cout << "\n[C++] FPGA output: First 48 values:" << std::endl;
        for (int i = 0; i < 48; ++i) {
            std::cout << dst_mem[i] << " ";
        }
    std::cout << std::endl;

    std::cout << "\n[C++] FPGA output: Second 48 values:" << std::endl;
        for (int i = 48; i < 96; ++i) {
            std::cout << dst_mem[i] << " ";
        }
    std::cout << std::endl; 

    // Make sure destination matches the source + 1 (the vFPGA logic in perf_local adds 1 to every 32-bit element, i.e. integer)
    //for (int i = 0; i < sg.local.src_len / sizeof(int); i++) {
    //    assert(src_mem[i] + 1 == dst_mem[i]); 
    //}

    return bench.getAvg();
}

int main(int argc, char *argv[])  {
    // CLI arguments
    unsigned int min_size, max_size, n_runs;
    boost::program_options::options_description runtime_options("Coyote Perf GPU Options");
    runtime_options.add_options()
        ("runs,r", boost::program_options::value<unsigned int>(&n_runs)->default_value(1), "Number of times to repeat the test") // 100
        ("min_size,x", boost::program_options::value<unsigned int>(&min_size)->default_value(49152), "Starting (minimum) transfer size")   // smaller than 192 bytes doesn't make sense as we need 3x16 ints per sample; original: 64, changed to: 256
        ("max_size,X", boost::program_options::value<unsigned int>(&max_size)->default_value(49152), "Ending (maximum) transfer size");   // original: 4 * 1024 * 1024
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);

    PR_HEADER("CLI PARAMETERS:");
    std::cout << "Number of test runs: " << n_runs << std::endl;
    std::cout << "Starting transfer size: " << min_size << std::endl;
    std::cout << "Ending transfer size: " << max_size << std::endl << std::endl;

    // GPU memory will be allocated on the GPU set using hipSetDevice(...)
    if (hipSetDevice(DEFAULT_GPU_ID)) { throw std::runtime_error("Couldn't select GPU!"); }

    // Obtain a Coyote thread and allocate memory
    // Note, the only difference from Example 1 is the way memory is allocated
    std::unique_ptr<coyote::cThread<std::any>> coyote_thread(new coyote::cThread<std::any>(DEFAULT_VFPGA_ID, getpid(), 0));

    unsigned int allocated_size = 4 * 1024 * 1024;  // If allocated > 4’194’304 Bytes (4MiB), the FPGA output will be all 0s as src_mem & dst_mem pointers will be in the same "window"

    int *src_mem = (int *) coyote_thread->getMem({coyote::CoyoteAlloc::GPU, allocated_size});  //max_size
    int *dst_mem = (int *) coyote_thread->getMem({coyote::CoyoteAlloc::GPU, allocated_size});  //max_size
    if (!src_mem || !dst_mem) {  throw std::runtime_error("Couldn't allocate memory"); }
    if (!src_mem || !dst_mem) { throw std::runtime_error("Could not allocate memory; exiting..."); }

    // Benchmark sweep of latency and throughput, with functional verification (correctness) in run_bench
    coyote::sgEntry sg;
    PR_HEADER("PERF GPU");
    unsigned int curr_size = min_size;
    sg.local = {.src_addr = src_mem, .dst_addr = dst_mem};
    while(curr_size <= max_size) {
        std::cout << "Size: " << std::setw(8) << curr_size << "; ";
        sg.local.src_len = curr_size; sg.local.dst_len = curr_size; 

        double throughput_time = run_bench(coyote_thread, sg, src_mem, dst_mem, N_THROUGHPUT_REPS, n_runs);
        double throughput = ((double) N_THROUGHPUT_REPS * (double) curr_size) / (1024.0 * 1024.0 * throughput_time * 1e-9);
        std::cout << "Average throughput: " << std::setw(8) << throughput << " MB/s; ";
        
        double latency_time = run_bench(coyote_thread, sg, src_mem, dst_mem, N_LATENCY_REPS, n_runs);
        std::cout << "Average latency: " << std::setw(8) << latency_time / 1e3 << " us" << std::endl;

        curr_size *= 2;
    }

    // Note, how there is no memory de-allocation, since the memory was allocated using coyote_thread->getMem(...)
    // A Coyote thread always keeps track of the memory it allocated and internally handles de-allocation
    return EXIT_SUCCESS;
}
