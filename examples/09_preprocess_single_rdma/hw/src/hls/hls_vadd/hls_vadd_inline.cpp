#include "hls_vadd.hpp"

void hls_vadd (
    hls::stream<axi_s> &axi_in,
    hls::stream<axi_s> &axi_out
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis register port=axi_in name=s_axi_in
    #pragma HLS INTERFACE axis register port=axi_out name=m_axi_out

    hls::stream<data_t> stream_load;
    #pragma HLS stream variable=stream_load depth=64

    hls::stream<data_t> stream_processed;
    #pragma HLS stream variable=stream_processed depth=64

    #pragma HLS dataflow

    LoadData(axi_in, stream_load);
    ProcessPackets(stream_load, stream_processed);
    StoreData(stream_processed, axi_out);
}

