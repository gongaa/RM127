#ifndef PRECODER_HPP_
#define PRECODER_HPP_

#include <string>
#include <vector>
#include "Util/Util.hpp"
using namespace std;

class Precoder
{
public:
    // Rate Profiling method: choose between Reed-Muller and Polar (channel)
    // essentially the same as construction problem
    CONSTRUCTION con = CONSTRUCTION::RM;
    vector<bool> frozen_bits;

    int K = 0;
    int N = 0;

    vector<int> V_N;

    explicit Precoder();
    Precoder(const int K, const int N, CONSTRUCTION con) : K(K), N(N), V_N(N), con(con), frozen_bits(N) {
        vector<bool> stab_info_bits(N, 0);
        construct_frozen_bits(con, N, K, K, frozen_bits, stab_info_bits);
    }

    virtual ~Precoder() = default;

    virtual void precode(const vector<int>& V_K, vector<int>& U_N) = 0;
    // use CNOT, so that the code in the X basis can be found
    virtual void precode_CNOT(const vector<int>& V_K, vector<int>& U_N) = 0; 

    virtual int solve_u(const vector<int>& V_N, int i) = 0; 
    virtual void decode(const vector<int>& U_N, vector<int>& V_K) = 0;

    virtual int get_K() { return K; }
    virtual int get_N() { return N; }

    virtual void rate_profiling(const vector<int>& V_K) {
        int j = 0;
        for (int i = 0; i < N; i++) {
            if (frozen_bits[i]) V_N[i] = 0;
            else V_N[i] = V_K[j++];
        }
    }

    Precoder* build() const;

};

#endif // PRECODER_HPP
