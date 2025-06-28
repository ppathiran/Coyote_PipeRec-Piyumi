#include "hls_vadd.hpp"

void hls_vadd (
    hls::stream<axi_s> &axi_dense,
    hls::stream<axi_s> &axi_sp0,
    hls::stream<axi_s> &axi_sp1,
    hls::stream<axi_s> &axi_out
) {
    // A free-runing kernel; no control interfaces needed to start the operation
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Specify that the input/output signals are AXI streams (axis)
    #pragma HLS INTERFACE axis register port=axi_dense name=s_axi_dense
    #pragma HLS INTERFACE axis register port=axi_sp0   name=s_axi_sp0
    #pragma HLS INTERFACE axis register port=axi_sp1   name=s_axi_sp1
    #pragma HLS INTERFACE axis register port=axi_out name=m_axi_out

    hls::stream<data_t> stream_load;
#pragma HLS stream variable=stream_load depth=64

    hls::stream<data_t> stream_zero;
#pragma HLS stream variable=stream_zero depth=64

    hls::stream<data_t> stream_log;
#pragma HLS stream variable=stream_log depth=64

    hls::stream<data_t> stream_mod;
#pragma HLS stream variable=stream_mod depth=64

//#pragma HLS dataflow

    // LoadData: Read from the AXI stream and write to the internal stream
    // Dense_NegsToZero: Read from the internal stream and write to the internal stream
    // Dense_Log: Read from the internal stream and write to the internal stream
    // Sparse_HexToIntMod: Read from the internal stream and write to the AXI streams
    // StoreData: Read from the internal stream and write to the AXI stream

//    LoadData(axi_in, stream_load);

//    Dense_NegsToZero(stream_load, stream_zero);

//    Dense_Log(stream_zero, stream_log);
 
    // Sparse_HexToIntMod(stream_log, stream_mod); 
//    Dense_Log(stream_log, stream_mod); 
 
//    StoreData(stream_mod, axi_out);    
//}

    static ap_uint<2> phase = 0;

#pragma HLS PIPELINE II=1
    while (true) {
        if (phase == 0 && !axi_dense.empty()) {
            axi_out.write(axi_dense.read()); phase = 1;
        } else if (phase == 1 && !axi_sp0.empty()) {
            axi_out.write(axi_sp0.read());   phase = 2;
        } else if (phase == 2 && !axi_sp1.empty()) {
            axi_out.write(axi_sp1.read());   phase = 0;
        }
    }    
}
