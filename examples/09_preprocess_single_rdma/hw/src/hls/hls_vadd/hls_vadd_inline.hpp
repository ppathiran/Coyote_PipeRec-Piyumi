#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>

#define AXI_DATA_BITS 512
typedef ap_axiu<AXI_DATA_BITS, 0, 0, 0> axi_s;

struct data_t {
    ap_uint<AXI_DATA_BITS> data;
    ap_uint<AXI_DATA_BITS/8> keep;
    bool last;
};

#define FLOAT_BITS 32
#define NUM_FLOATS AXI_DATA_BITS / FLOAT_BITS

typedef union {
    float float32;
    uint32_t uint32;
} conv;

void hls_vadd (
    hls::stream<axi_s> &axi_in,
    hls::stream<axi_s> &axi_out
);

void LoadData(
    hls::stream<axi_s> &axi_in,
    hls::stream<data_t> &stream_out
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    if (!axi_in.empty()) {
        axi_s input_data = axi_in.read();
        data_t output_data;
        output_data.data = input_data.data;
        output_data.keep = input_data.keep;
        output_data.last = input_data.last;
        stream_out.write(output_data);
    }
}

void StoreData(
    hls::stream<data_t> &stream_in,
    hls::stream<axi_s> &axi_out
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    if (!stream_in.empty()) {
        data_t input_data = stream_in.read();
        axi_s output_data;
        output_data.data = input_data.data;
        output_data.keep = input_data.keep;
        output_data.last = input_data.last;
        axi_out.write(output_data);
    }
}

void ProcessPackets(
    hls::stream<data_t> &stream_in,
    hls::stream<data_t> &stream_out
) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1

    static unsigned int packet_counter = 0;

    while (true) {
#pragma HLS PIPELINE II=1

        if (!stream_in.empty()) {
            data_t pkt = stream_in.read();
            ap_uint<512> in_data = pkt.data;
            ap_uint<512> out_data;

            if (packet_counter % 3 == 0) {
                // Dense_NegsToZero
                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    int tmp_val = in_data(32*i+31, 32*i);
                    tmp_val = (tmp_val < 0) ? 0 : tmp_val;
                    in_data(32*i+31, 32*i) = tmp_val;
                }

                // Dense_Log
                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    int tmp_val = in_data(32*i+31, 32*i);
                    conv conv_val;
                    conv_val.float32 = hls::logf(tmp_val + 1);
                    out_data(32*i+31, 32*i) = conv_val.uint32;
                }

            } else {
                // Sparse_HexToIntMod
                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    ap_uint<32> tmp_input = in_data(32*i+31, 32*i);
                    ap_uint<32> tmp_output = tmp_input & 0x3FF; // % 1024
                    out_data(32*i+31, 32*i) = tmp_output;
                }
            }

            data_t out_pkt;
            out_pkt.data = out_data;
            out_pkt.keep = pkt.keep;
            out_pkt.last = pkt.last;
            stream_out.write(out_pkt);

            packet_counter++;
        }
    }
}

