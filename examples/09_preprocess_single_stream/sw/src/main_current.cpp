#include <any>
#include <iostream>
#include <cstdlib>
#include <iomanip>

// AMD GPU management & run‑time libraries
#include <hip/hip_runtime.h>

// External library for easier parsing of CLI arguments
#include <boost/program_options.hpp>

// Coyote‑specific includes
#include "cBench.hpp"
#include "cThread.hpp"

// -----------------------------------------------------------------------------
// New sample‑related constants
// -----------------------------------------------------------------------------
#define SAMPLE_INTS   48                                  // 48 × int per sample
#define SAMPLE_BYTES (SAMPLE_INTS * sizeof(int))          // 192 bytes/sample

// Benchmark parameters (unchanged)
#define N_LATENCY_REPS    1
#define N_THROUGHPUT_REPS 64

#define DEFAULT_GPU_ID   0
#define DEFAULT_VFPGA_ID 0

// -----------------------------------------------------------------------------
// Helper function (unchanged except for stricter size check)
// -----------------------------------------------------------------------------
double run_bench(
    std::unique_ptr<coyote::cThread<std::any>>& coyote_thread,
    coyote::sgEntry& sg,
    int* src_mem, int* dst_mem,
    uint transfers, uint n_runs
) {
    // Make sure we transfer an integer number of samples
    assert(sg.local.src_len == sg.local.dst_len);
    assert((sg.local.src_len % SAMPLE_BYTES) == 0);

    // Randomize source, clear destination
    for (size_t i = 0; i < sg.local.src_len / sizeof(int); ++i) {
        src_mem[i] = rand() % 1024 - 512;
        dst_mem[i] = 0;
    }

    std::cout << "\n[C++] Data generated (first sample):" << std::endl;
        for (int i = 0; i < 48; ++i) {
            std::cout << src_mem[i] << " ";
        }
        std::cout << std::endl;

    auto prep_fn = [&]() { coyote_thread->clearCompleted(); };

    coyote::cBench bench(n_runs);
    auto bench_fn = [&]() {
        // Queue transfers
        for (uint i = 0; i < transfers; ++i)
            coyote_thread->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg);

        // Wait for completion
        while (coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) != transfers) {}
    };
    bench.execute(bench_fn, prep_fn);

    std::cout << "\n[C++] FPGA output: First sample:" << std::endl;
        for (int i = 0; i < 100; ++i) {
            std::cout << dst_mem[i] << " ";
        }
        std::cout << std::endl;
    
    return bench.getAvg();
}

int main(int argc, char* argv[]) {
    // -------------------------------------------------------------------------
    // CLI: now interprets min/max **samples** instead of bytes
    // -------------------------------------------------------------------------
    unsigned int min_samples, max_samples, n_runs;
    boost::program_options::options_description opts("Coyote Perf GPU Options");
    opts.add_options()
        ("runs,r",      boost::program_options::value<unsigned int>(&n_runs)      ->default_value(100), "Number of repetitions")
        ("min_samples,x", boost::program_options::value<unsigned int>(&min_samples)->default_value(64),  "Starting number of samples")
        ("max_samples,X", boost::program_options::value<unsigned int>(&max_samples)->default_value(8 * 1024), "Ending number of samples");   // 64 * 1024
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opts), vm);
    boost::program_options::notify(vm);

    PR_HEADER("CLI PARAMETERS:");
    std::cout << "Runs                : " << n_runs       << '\n'
              << "Starting #samples   : " << min_samples  << '\n'
              << "Ending   #samples   : " << max_samples  << "\n\n";

    // -------------------------------------------------------------------------
    // GPU & Coyote thread setup (unchanged)
    // -------------------------------------------------------------------------
    if (hipSetDevice(DEFAULT_GPU_ID))
        throw std::runtime_error("Couldn't select GPU!");

    std::unique_ptr<coyote::cThread<std::any>> coyote_thread(
        new coyote::cThread<std::any>(DEFAULT_VFPGA_ID, getpid(), 0));

    // Allocate GPU buffers big enough for the largest sample count
    const size_t max_bytes = static_cast<size_t>(max_samples) * SAMPLE_BYTES;

    unsigned int max_size = 12 * 1024 * 1024;      //4 * 1024 * 1024    works with 12

    int* src_mem = static_cast<int*>(coyote_thread->getMem({coyote::CoyoteAlloc::GPU, max_size}));
    int* dst_mem = static_cast<int*>(coyote_thread->getMem({coyote::CoyoteAlloc::GPU, max_size}));


    if (!src_mem || !dst_mem)
        throw std::runtime_error("Could not allocate GPU memory via Coyote.");

    // -------------------------------------------------------------------------
    // Benchmark loop: sweep over sample counts
    // -------------------------------------------------------------------------
    coyote::sgEntry sg;
    sg.local = {.src_addr = src_mem, .dst_addr = dst_mem};

    PR_HEADER("PERF GPU (per sample stream)");
    for (unsigned int curr_samples = min_samples;
         curr_samples <= max_samples;
         curr_samples *= 2)
    {
        const size_t bytes_this_run = static_cast<size_t>(curr_samples) * SAMPLE_BYTES;
        sg.local.src_len = sg.local.dst_len = bytes_this_run;

        std::cout << "Samples: " << std::setw(8) << curr_samples
                  << "  (Bytes: " << std::setw(9) << bytes_this_run << ")  ";

        // Throughput
        double tput_time = run_bench(coyote_thread, sg, src_mem, dst_mem,
                                     N_THROUGHPUT_REPS, n_runs);
        double tput_MBps = (static_cast<double>(N_THROUGHPUT_REPS) * bytes_this_run) /
                           (1024.0 * 1024.0 * tput_time * 1e-9);
        std::cout << "Avg throughput: " << std::setw(10) << tput_MBps << " MB/s; ";

        // Latency
        double lat_time = run_bench(coyote_thread, sg, src_mem, dst_mem,
                                    N_LATENCY_REPS, n_runs);
        std::cout << "Avg latency: " << std::setw(8) << lat_time / 1e3 << " µs\n";
    }

    // No explicit de‑allocation needed (handled by cThread)
    return EXIT_SUCCESS;
}

