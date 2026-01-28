#ifndef SPARSE_PRECODER_HPP_
#define SPARSE_PRECODER_HPP_

#include "Precoder.hpp"
#include <Eigen/Sparse>

using namespace std;
// #define HAMMING
class Sparse_precoder: public Precoder
{
    // specified by a convolution kernel
    // encoding matrix is an upper triangular Toeplitz matrix
protected:
    Eigen::SparseMatrix<int> mat; // Precoding matrix, diagonal omitted
  
public:
    Sparse_precoder(const int K, const int N, CONSTRUCTION con, Eigen::SparseMatrix<int> mat) : Precoder(K, N, con), mat(mat) { }
    virtual ~Sparse_precoder() = default;
    void precode(const vector<int>& V_K, vector<int>& U_N);
    void precode_CNOT(const vector<int>& V_K, vector<int>& U_N); 
    void precode_CNOT_reverse(const vector<int>& V_K, vector<int>& U_N);
    void decode(const vector<int>& U_N, vector<int>& V_K);
    int solve_u(const vector<int>& V_N, int i); 

    // static bool is_codeword(const vector<int>& X_N);
};
      
#endif // SPARSE_PRECODER_HPP_