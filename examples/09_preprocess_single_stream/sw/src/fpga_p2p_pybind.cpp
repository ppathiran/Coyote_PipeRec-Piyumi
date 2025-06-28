#include <any>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <ctime>

// AMD GPU management & run-time libraries
#include <hip/hip_runtime.h>

// Coyote-specific includes
#include "cBench.hpp"
#include "cThread.hpp"

// for pybind11 wrapper
#include <memory>
#include <chrono>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Constants
#define N_LATENCY_REPS 1
#define N_THROUGHPUT_REPS 1
#define DEFAULT_GPU_ID 0
#define DEFAULT_VFPGA_ID 0

namespace py = pybind11;

class FpgaP2PWrapper {
public:
    int batch_size;
    int num_elements;
    unsigned int size;
    unsigned int n_runs;

    std::unique_ptr<coyote::cThread<std::any>> coyote_thread;
    int *src_mem;
    int *dst_mem[2];
    coyote::sgEntry sg;

    FpgaP2PWrapper(int batch_size_input) {
        batch_size = batch_size_input;
        num_elements = batch_size * 48;
        size = num_elements * sizeof(int);
        n_runs = 1;

        if (hipSetDevice(DEFAULT_GPU_ID)) {
            throw std::runtime_error("Couldn't select GPU!");
        }

        coyote_thread = std::make_unique<coyote::cThread<std::any>>(DEFAULT_VFPGA_ID, getpid(), 0);

        src_mem = static_cast<int *>(coyote_thread->getMem({coyote::CoyoteAlloc::GPU, size}));
        dst_mem[0] = static_cast<int *>(coyote_thread->getMem({coyote::CoyoteAlloc::GPU, size}));
        dst_mem[1] = static_cast<int *>(coyote_thread->getMem({coyote::CoyoteAlloc::GPU, size}));

        sg.local = {.src_addr = src_mem, .src_len = size, .dst_addr = dst_mem[0], .dst_len = size};
    }

    void generate_batch_data() {
        srand(time(NULL));
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
                dst_mem[0][base_idx + i] = 0;
                dst_mem[1][base_idx + i] = 0;
            }
        }

        std::cout << "\n[C++] Batch data generated (first sample):" << std::endl;
        for (int i = 0; i < 48; ++i) {
            std::cout << src_mem[i] << " ";
        }
        std::cout << std::endl;
    }

    double run_bench(uint transfers, int buffer_idx) {
        assert(sg.local.src_len == sg.local.dst_len);

        coyote::cBench bench(n_runs);

        auto prep_fn = [&]() {
            coyote_thread->clearCompleted();
        };

        auto bench_fn = [&]() {
            for (int i = 0; i < transfers; i++) {
                coyote_thread->invoke(coyote::CoyoteOper::LOCAL_TRANSFER, &sg);
                std::cout << "[C++] Called invoke on FPGA kernel per transfer" << std::endl;
            }
	    std::cout << "[C++] Called invoke on FPGA kernel afer for-loop" << std::endl;
            while (coyote_thread->checkCompleted(coyote::CoyoteOper::LOCAL_TRANSFER) != transfers) {}
        };

        bench.execute(bench_fn, prep_fn);

        std::cout << "\n[C++] GPU Pointer: " << static_cast<void*>(dst_mem[buffer_idx]) << std::endl;

        std::cout << "\n[C++] FPGA output: First sample of the current batch:" << std::endl;
        for (int i = 0; i < 48; ++i) {
            std::cout << dst_mem[buffer_idx][i] << " ";
        }
        std::cout << std::endl;

        std::cout << "\n[C++] FPGA output: Last sample of the current batch:" << std::endl;
        int last_sample_base_idx = (batch_size - 1) * 48;
        for (int i = 0; i < 48; ++i) {
            std::cout << dst_mem[buffer_idx][last_sample_base_idx + i] << " ";
        }
        std::cout << std::endl;

        return bench.getAvg();
    }

    void preprocess_single_batch(int buffer_idx) {
        PR_HEADER("CLI PARAMETERS:");
        std::cout << "Transfer size: " << size << std::endl << std::endl;

        sg.local.dst_addr = dst_mem[buffer_idx];

        double latency_time = run_bench(N_THROUGHPUT_REPS, buffer_idx);
        double throughput = ((double)N_THROUGHPUT_REPS * (double)size) / (1024.0 * 1024.0 * latency_time * 1e-9);
        std::cout << "\n[C++] Preprocessing + Transfer Latency: " << std::setw(8) << latency_time / 1e3 << " us" << std::endl;
        std::cout << "[C++] Preprocessing + Transfer Throughput: " << std::setw(8) << throughput << " MB/s" << std::endl;
    }

    uint64_t get_dst_ptr(int buffer_idx) {
        return reinterpret_cast<uint64_t>(dst_mem[buffer_idx]);
    }
};

PYBIND11_MODULE(fpga_p2p_pybind, m) {
    py::class_<FpgaP2PWrapper>(m, "FpgaP2PWrapper")
        .def(py::init<int>())
        .def("get_dst_ptr", &FpgaP2PWrapper::get_dst_ptr, py::arg("buffer_idx"))
        .def("preprocess_single_batch", &FpgaP2PWrapper::preprocess_single_batch, py::arg("buffer_idx"))
        .def("generate_batch_data", &FpgaP2PWrapper::generate_batch_data);
}

