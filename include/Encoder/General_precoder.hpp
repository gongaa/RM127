#ifndef GENERAL_PRECODER_HPP_
#define GENERAL_PRECODER_HPP_

#include "Precoder.hpp"
#include <Eigen/Dense>

using namespace std;
// #define HAMMING
class General_precoder: public Precoder
{
    // specified by a convolution kernel
    // encoding matrix is an upper triangular Toeplitz matrix
protected:
    Eigen::MatrixXi mat; // Precoding matrix
  
public:
    General_precoder(const int K, const int N, CONSTRUCTION con, Eigen::MatrixXi mat) : Precoder(K, N, con), mat(mat) { }
    virtual ~General_precoder() = default;
    void precode(const vector<int>& V_K, vector<int>& U_N);
    void precode_CNOT(const vector<int>& V_K, vector<int>& U_N); 
    void decode(const vector<int>& U_N, vector<int>& V_K);
    int solve_u(const vector<int>& V_N, int i); 

    // static bool is_codeword(const vector<int>& X_N);
};
      
#endif // GENERAL_PRECODER_HPP_