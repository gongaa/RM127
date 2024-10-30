#include "Simulation/Simulation.hpp"
#include <fstream>
#include <sstream>

int simulation_punctured_RM(int r, int num_total, double p, int list_size, int seed) {
    // r=4 or 2 for [[127,1,7]]
    // r=3 for [[127,1,15]]
    int m = 7;
    Encoder_RM* encoder = new Encoder_RM(m, r);
    int K = encoder->get_K(), N = encoder->get_N();
    vector<bool> frozen_bits(N, 1);
    vector<int> bin(m, 0);
    int weight, K_cnt = 0;
    for (int i = 0; i < N; i++) {
        decimal2binary(i, bin);
        weight = count_weight(bin);
        if (weight >= (m-r)) { frozen_bits[i] = 0; K_cnt++; }
    }
    // Decoder_RM_SCL* SCL_decoder = new Decoder_RM_SCL(m, r, list_size);
    Decoder_polar_SCL* SCL_decoder = new Decoder_polar_SCL(K, N, list_size, frozen_bits);
    cerr << "For m=" << m << ", r="<< r << ", K=" << K << ", self-counted K=" << K_cnt << ", N=" << N << endl;
    cerr << "List size=" << list_size << endl;
    Channel_BSC* chn_bsc = new Channel_BSC(N-1, p, seed);

    vector<int> info_bits(K, 1);
    vector<int> codeword(N, 0);
    vector<int> noisy_codeword(N, 0);
    vector<double> llr_noisy_codeword(N, 0);
    vector<int> SCL_denoised_codeword(N, 0);
    vector<int> encoded_codeword(N, 0);
    int num_flips = 0, SCL_num_flips = 0, SCL_num_err = 0, class_bit = 0;

    for (int k = 0; k < num_total; k++) {
        generate_random(K, info_bits.data());
        encoder->encode(info_bits.data(), codeword.data(), 0);
        num_flips = chn_bsc->add_noise(codeword.data(), noisy_codeword.data(), 0);
        for (int i = 0; i < N-1; i++)
            llr_noisy_codeword[i] = noisy_codeword[i] ? -log((1-p)/p) : log((1-p)/p); // 0 -> 1.0; 1 -> -1.0
        llr_noisy_codeword[N-1] = 0; // last bit is punctured

        SCL_decoder->decode(llr_noisy_codeword.data(), SCL_denoised_codeword.data());
        SCL_num_flips = count_flip(N-1, noisy_codeword, SCL_denoised_codeword);
        encoder->encode_mm_code(SCL_denoised_codeword.data(), encoded_codeword.data(), N);
        // class_bit = SCL_decoder->get_last_info_bit();
        // cerr << "SCL: " << class_bit << ". encoded: " << encoded_codeword[N-1] << ". info bit: " << info_bits[K-1] << endl;
        // if (class_bit != info_bits[K-1]) {
        if (encoded_codeword[N-1] != info_bits[K-1]) {
            // cerr << "num of flips " << num_flips << "; SCL flips " << SCL_num_flips << endl;
            // cerr << "differ by " << count_flip(N-1, codeword, SCL_denoised_codeword) << endl;
            SCL_num_err++; 
        }
    }
    cerr << "#logical errors: " << SCL_num_err << "/" << num_total << endl;
    cerr << "logical error rate: " << (double)SCL_num_err / num_total << endl;


    return 0;
}