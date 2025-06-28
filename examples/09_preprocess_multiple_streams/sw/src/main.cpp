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
#define N_THROUGHPUT_REPS 64

#define DEFAULT_GPU_ID 0
#define DEFAULT_VFPGA_ID 0
#include <array>      //added

//modified run_bench to send 3 AXI streams
double run_bench_triple(
    std::unique_ptr<coyote::cThread<std::any>>& cyt,
    std::array<coyote::sgEntry,3>&              sg,
    std::array<int*,3>&                         src,
    std::array<int*,3>&                         dst,
    uint                                        transfers,
    uint                                        n_runs)
{
    
    for (int s=0; s<3; ++s){
        size_t n_int = sg[s].local.src_len / sizeof(int);
        for (size_t i=0;i<n_int;i++){ src[s][i] = rand()%1024 - 512; dst[s][i] = 0; }
    }

    auto  prep_fn  = [&](){ cyt->clearCompleted(); };

    coyote::cBench bench(n_runs);
    auto  bench_fn = [&](){
        for (uint rep=0; rep<transfers; ++rep){
            cyt->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg[0]);
            cyt->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg[1]);
            cyt->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg[2]);
	    std::cout << "Called invoke()"; //debug log
        }

	std::cout << "Wait for transfer completion"; //debug log
        while (cyt->checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) 
               < transfers*3) {}
    };

    bench.execute(bench_fn, prep_fn);
    return bench.getAvg();
}


int main(int argc,char* argv[])
{
    // CLI arguments
    unsigned int min_size, max_size, n_runs;
    boost::program_options::options_description runtime_options("Coyote Perf GPU Options");
    runtime_options.add_options()
        ("runs,r", boost::program_options::value<unsigned int>(&n_runs)->default_value(100), "Number of times to repeat the test")
        ("min_size,x", boost::program_options::value<unsigned int>(&min_size)->default_value(64), "Starting (minimum) transfer size")
        ("max_size,X", boost::program_options::value<unsigned int>(&max_size)->default_value(4 * 1024 * 1024), "Ending (maximum) transfer size");
    boost::program_options::variables_map command_line_arguments;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, runtime_options), command_line_arguments);
    boost::program_options::notify(command_line_arguments);

    PR_HEADER("CLI PARAMETERS:");
    std::cout << "Number of test runs: " << n_runs << std::endl;
    std::cout << "Starting transfer size: " << min_size << std::endl;
    std::cout << "Ending transfer size: " << max_size << std::endl << std::endl;

    if (hipSetDevice(DEFAULT_GPU_ID)) 
        throw std::runtime_error("Couldn't select GPU!");

    std::unique_ptr<coyote::cThread<std::any>> cyt(
        new coyote::cThread<std::any>(DEFAULT_VFPGA_ID, getpid(), 0));

    //allocate 3 src/dst buffers
    std::array<int*,3> src_mem, dst_mem;
    for (int s=0; s<3; ++s){
        src_mem[s] = (int*)cyt->getMem({coyote::CoyoteAlloc::GPU, max_size});
        dst_mem[s] = (int*)cyt->getMem({coyote::CoyoteAlloc::GPU, max_size});
        if (!src_mem[s] || !dst_mem[s]) 
            throw std::runtime_error("Couldn't allocate GPU buffers");
    }

    //3 SG descriptors 
    std::array<coyote::sgEntry,3> sg;
    for (int s=0; s<3; ++s)
        sg[s].local = { .src_addr = src_mem[s],
                .src_len  = 0,
                .dst_addr = dst_mem[s],
                .dst_len  = 0 };


    PR_HEADER("PERF‑GPU triple‑stream");
    unsigned curr_size = min_size;
    while (curr_size <= max_size) {
        std::cout << "Size: " << std::setw(8) << curr_size << "; ";

        for (int s=0; s<3; ++s){
            sg[s].local.src_len = curr_size;
            sg[s].local.dst_len = curr_size;
        }

        double thr_time = run_bench_triple(cyt, sg, src_mem, dst_mem,
                                           N_THROUGHPUT_REPS, n_runs);
        double throughput = (double)N_THROUGHPUT_REPS * 3.0 * 
                            (double)curr_size / (1024.0*1024.0*thr_time*1e-9);
        std::cout << "Avg throughput: " << std::setw(8) << throughput << " MB/s; ";

        double lat_time = run_bench_triple(cyt, sg, src_mem, dst_mem,
                                           N_LATENCY_REPS, n_runs);
        std::cout << "Avg latency: " << std::setw(8) << lat_time/1e3 << " us\n";

        curr_size *= 2;
    }
    return EXIT_SUCCESS;
}

