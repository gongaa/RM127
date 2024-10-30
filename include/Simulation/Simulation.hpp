#ifndef SIMULATION_HPP_
#define SIMULATION_HPP_
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>
#include "Encoder/Encoder_RM.hpp"
#include "Channel/Channel.hpp"
#include "Encoder/Encoder_polar.hpp"
#include "Decoder/Decoder_polar.hpp"
#include "Util/Util.hpp"

// #define CHN_AWGN
// #define VERBOSE
// #define COPY_LIST

// Reed-Muller
int simulation_punctured_RM(int r, int num_total, double p, int list_size, int seed);

#endif