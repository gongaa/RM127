#ifndef CONVOLUTION_PRECODER_HPP_
#define CONVOLUTION_PRECODER_HPP_

#include "Precoder.hpp"

using namespace std;
// #define HAMMING
class Convolution_precoder: public Precoder
{
    // specified by a convolution kernel
    // encoding matrix is an upper triangular Toeplitz matrix
protected:
    vector<int> c = {1,0,1,1,0,1,1}; // convolution kernel
    int ks = 7; // kernel size
  
public:
    Convolution_precoder(const int K, const int N, CONSTRUCTION con) : Precoder(K, N, con) {}
    virtual ~Convolution_precoder() = default;
    void precode(const vector<int>& V_K, vector<int>& U_N);
    void precode_CNOT(const vector<int>& V_K, vector<int>& U_N); 
    void precode_CNOT_reverse(const vector<int>& V_K, vector<int>& U_N); 
    void decode(const vector<int>& U_N, vector<int>& V_K);
    int solve_u(const vector<int>& V_N, int i); 

    // static bool is_codeword(const vector<int>& X_N);
};
      
#endif // CONVOLUTION_PRECODER_HPP_