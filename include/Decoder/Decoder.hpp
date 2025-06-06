#ifndef DECODER_HPP_
#define DECODER_HPP_

#include <vector>
#include <cmath>
#include <functional>
using namespace std;
 
class Decoder
{
public:
    const int K, N;
    // lambdas are LLR update rules
    vector<function<double(const vector<double> &LLRs, const vector<int> &bits)>> my_lambdas;

public:
    explicit Decoder(const int K, const int N);
    static double phi(const double& mu, const double& lambda, const int& u); // path metric update function
    static void f_plus(const double* LLR_fst, const double* LLR_snd, const int size, double* LLR_new);
    static void f_minus(const double* LLR_fst, const double* LLR_snd, const int* bits, const int size, double* LLR_new);
    void f_plus_depolarize(const vector<double>* p_fst, const vector<double>* p_snd, 
        const int size, vector<double>* p_new);
    void f_minus_depolarize(const vector<double>* p_fst, const vector<double>* p_snd, 
        const vector<vector<int>>& u, const int size, vector<double>* p_new);
    double phi_depolarize(const double& mu, const vector<double>& p, const int& ux, const int& uz);
};

#endif /* DECODER_HPP_ */
