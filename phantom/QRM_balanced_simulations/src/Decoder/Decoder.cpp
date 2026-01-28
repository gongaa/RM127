#include "Decoder/Decoder.hpp"
#include <iostream>
#include <cassert>
#include <immintrin.h>
#include <cmath>
using namespace std;
#define USE_APPROXIMATION

vector<function<double(const vector<double> &LLRs, const vector<int> &bits)>> my_lambdas =  {
    [](const vector<double> &LLRs, const vector<int> &bits) -> double
    {   // the hardware-efficient f- function
        auto sign = std::signbit(LLRs[0]) ^ std::signbit(LLRs[1]);
        auto abs0 = std::abs(LLRs[0]);
        auto abs1 = std::abs(LLRs[1]);
        auto min = std::min(abs0, abs1);
        return sign ? -min : min;
    },
    [](const vector<double> &LLRs, const vector<int> &bits) -> double
    {   // the f+ function
        return ((bits[0] == 0) ? LLRs[0] : -LLRs[0]) + LLRs[1];
    }
};

// auto Decoder::lambdas = vector<function<double(const vector<double> &LLRs, const vector<int> &bits)>>(my_lambdas);

void Decoder::f_plus(const double* LLR_fst, const double* LLR_snd, const int size, double* LLR_new)
{
    for (int i = 0; i < size; i++) {
#ifdef USE_APPROXIMATION
        auto sign = signbit(LLR_fst[i]) ^ signbit(LLR_snd[i]);
        auto abs0 = abs(LLR_fst[i]);
        auto abs1 = abs(LLR_snd[i]);
        auto min = std::min(abs0, abs1);
        LLR_new[i] = sign ? -min : min;
#else
        LLR_new[i] = log(exp(LLR_fst[i] + LLR_snd[i]) + 1) - log(exp(LLR_fst[i]) + exp(LLR_snd[i]));
#endif // USE_APPROXIMATION
    }
}

void Decoder::f_minus(const double* LLR_fst, const double* LLR_snd, const int* bits, const int size, double* LLR_new)
{
    for (int i = 0; i < size; i++) {
        LLR_new[i] = ((bits[i] == 0) ? LLR_fst[i] : -LLR_fst[i]) + LLR_snd[i];
    }
}

Decoder::Decoder(const int K, const int N) : K(K), N(N)
{
}

double Decoder::phi(const double& mu, const double& lambda, const int& u)
{   // path metric update function
    assert(!isnan(mu));
    assert(!isnan(lambda));
    double new_mu;
#ifdef USE_APPROXIMATION
    if (u == 0 && lambda < 0)
        new_mu = mu - lambda;
    else if (u != 0 && lambda > 0)
        new_mu = mu + lambda;
    else // if u = [1-sign(lambda)]/2 correct prediction
        new_mu = mu;
#else
    new_mu = mu + ((u == 0) ? log(1 + exp(-lambda)) : log(1 + exp(lambda)));
    assert(!isnan(new_mu));
#endif // USE_APPROXIMATION
    return new_mu;
}

void Decoder::f_plus_quaternary(const vector<double>* p_fst, const vector<double>* p_snd, 
    const int size, vector<double>* p_new)
{
    for (int i = 0; i < size; i++) {
        const vector<double>& p = p_fst[i];
        // for (auto pp : p) cerr << pp << " ";
        // cerr << endl;
        const vector<double>& q = p_snd[i];
        // for (auto pp : q) cerr << pp << " ";
        // cerr << endl;
        vector<double>& r = p_new[i];
        // TODO: change to convolution
        r[0] = p[0]*q[0] + p[1]*q[1] + p[2]*q[2] + p[3]*q[3]; // I = (0,0)
        r[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] + p[3]*q[2]; // X = (0,1)
        r[2] = p[0]*q[2] + p[1]*q[3] + p[2]*q[0] + p[3]*q[1]; // Z = (1,0)
        r[3] = p[0]*q[3] + p[1]*q[2] + p[2]*q[1] + p[3]*q[0]; // Y = (1,1)
        // Pr(u1=a, v1=b) = Pr(u1+u2=0, v1+v2=0) * Pr(u2=a, v2=b) 
        //                + Pr(u1+u2=0, v1+v2=1) * Pr(u2=a, v2=b+1)
        //                + Pr(u1+u2=1, v1+v2=0) * Pr(u2=a+1, v2=b)
        //                + Pr(u1+u2=1, v1+v2=1) * Pr(u2=a+1, v2=b+1)
    }
}

void Decoder::f_minus_quaternary(const vector<double>* p_fst, const vector<double>* p_snd, 
    const vector<vector<int>>& u, const int size, vector<double>* p_new)
{
    for (int i = 0; i < size; i++) {
        const vector<double>& p = p_fst[i];
        const vector<double>& q = p_snd[i];
        vector<double>& r = p_new[i];
        const int hat_v1 = u[i][0], hat_u1 = u[i][1];
        // Pr(u2=a, v2=b) = Pr(u1+u2=a+\hat{u}1, v1+v2=b+\hat{v}1) * Pr(u2=a, v2=b) / normalization
        // TODO: write this in convolution
        if (hat_u1==0) {
            if (hat_v1 == 0) { // ( \hat{u}1, \hat{v}1 ) = (0,0)
                r[0] = p[0]*q[0]; r[1] = p[1]*q[1]; r[2] = p[2]*q[2]; r[3] = p[3]*q[3];
            } else {           // ( \hat{u}1, \hat{v}1 ) = (0,1)
                r[0] = p[1]*q[0]; r[1] = p[0]*q[1]; r[2] = p[3]*q[2]; r[3] = p[2]*q[3];
            }
        } else {
            if (hat_v1 == 0) { // ( \hat{u}1, \hat{v}1 ) = (1,0)
                r[0] = p[2]*q[0]; r[1] = p[3]*q[1]; r[2] = p[0]*q[2]; r[3] = p[1]*q[3];
            } else {           // ( \hat{u}1, \hat{v}1 ) = (1,1)
                r[0] = p[3]*q[0]; r[1] = p[2]*q[1]; r[2] = p[1]*q[2]; r[3] = p[0]*q[3];
            }
        }
        double total = r[0] + r[1] + r[2] + r[3];
        r[0] /= total; r[1] /= total; r[2] /= total; r[3] /= total;
    }
}

double Decoder::phi_depolarize(const double& mu, const vector<double>& p, const int& u1, const int& v1)
{   // path metric update function
    // p = ( p_I, p_X, p_Z, p_Y ) := ( p_{(0,0)}, p_{(0,1)}, p_{(1,0)}, p_{1,1} )
    assert(!isnan(mu));
    // TODO: write this in LLR
    double new_mu = mu - log(p[2*u1 + v1]);
    assert(!isnan(new_mu));
    return new_mu;
}