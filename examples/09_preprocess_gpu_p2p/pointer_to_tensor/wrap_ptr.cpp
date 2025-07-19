#include <torch/extension.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

// copy memory from raw GPU ptr into a PyTorch tensor
torch::Tensor copy_coyote_ptr(uint64_t ptr, int64_t numel, int device_id) {
    hipSetDevice(device_id);

    void* raw_ptr = reinterpret_cast<void*>(ptr);

    // debug info
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, raw_ptr);
    if (err != hipSuccess) {
        std::cerr << "[C++ Wrapper] hipPointerGetAttributes failed: "
                  << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipPointerGetAttributes failed");
    }
    //std::cerr << "[C++ Wrapper] Pointer is on device: " << attr.device << std::endl;

    // create a dst tensor on GPU with dtype uint32
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt32)
                       .device(torch::kCUDA, device_id);
    auto dest = torch::empty({numel}, options);

    // copy memory from raw_ptr to dest
    err = hipMemcpy(dest.data_ptr(), raw_ptr, numel * sizeof(uint32_t), hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        std::cerr << "[C++ Wrapper] hipMemcpy failed: "
                  << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
    }

    return dest;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_coyote_ptr", &copy_coyote_ptr, "Copy Coyote GPU pointer into PyTorch tensor");
}

