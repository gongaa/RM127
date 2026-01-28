#ifndef UTIL_HPP_
#define UTIL_HPP_
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
using namespace std;

template <typename T> inline void xor_vec(int N, vector<T>& a, const vector<T>& b, vector<T>& c) {
    for (int i = 0; i < N; i++)
        c[i] = a[i] ^ b[i];
}


template <typename T> inline int dot_product(int N, vector<T>& a, vector<T>& b) {
    int sum = 0;
    for(int i = 0; i < N; i++)
        if (a[i] && b[i]) sum++;
    return (sum % 2);
}

template <typename T> inline void bit_reversal(vector<T>& a) {
    T temp;
    int N = a.size();
    for (int i = 0; i < N/2; i++) {
        temp = a[i];
        a[i] = a[N-1-i];
        a[N-1-i] = temp;
    }
}

template <typename T> inline int count_flip(int N, vector<T>& a, vector<T>& b) {
    int cnt = 0;
    for (int i = 0; i < N; i++) 
        if (a[i] != b[i]) cnt++;
    return cnt;
}

template <typename T> inline int count_weight(const vector<T>& a) {
    int cnt = 0;
    for (auto i : a)
        if (i) cnt++;
    return cnt;
}

inline int wt(int N) {
    int result = 0;
    while (N) {
        result++;
        N &= N-1; // unset the least significant bit
    }
    return result;
}

inline double db2val(double x) {
  return exp(log(10.0) * x / 10.0);
}

template <typename T> inline void decimal2binary(const int& n, vector<T>& b)
{
    int x = n;
    for (int i = 0; x > 0; i++) {
        b[i] = x % 2;
        x >>= 1;
    }
}

template <typename T> inline int binary2decimal(const vector<T>& b, int size)
{
    int n = 0;
    for (int i = 0; i < size; i++) 
        if (b[i])
            n += (1 << i);
    return n;
}

template <typename T> 
ostream& operator<<(ostream& os, const vector<T>& v) 
{ 
    os << "[";
    for (int i = 0; i < v.size(); ++i) { 
        os << v[i]; 
        if (i != v.size() - 1) 
            os << ", "; 
    }
    os << "]\n";
    return os; 
}

inline void generate_random(int N, int *Y_N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(0.5);
 
    for (int i = 0; i < N; i++) {
        Y_N[i] = d(gen);
    }
}

inline void print_wt_dist(vector<int>& wt) {
    sort(wt.begin(), wt.end());
    int i = wt[0];
    int cnt = 1;
    for (int k = 1; k < wt.size(); k++) {
        if (wt[k] == i) cnt++;
        else {
            cerr << i << ":" << cnt << "  ";
            i = wt[k]; cnt = 1;
        }
    }
    cerr << i << ":" << cnt << endl;
}

inline void cal_wt_dist_prob(vector<int>& wt, double& p, const int& offset, const double& weight) {
    // wt should be sorted before calling this function
    int i = wt[0];
    int cnt = 1;
    for (int k = 1; k < wt.size(); k++) {
        if (wt[k] == i) cnt++;
        else {
            p += pow(weight, i-offset) * cnt;
            i = wt[k]; cnt = 1;
        }
    }
    p += pow(weight, i-offset) * cnt;
}

#endif // UTIL_HPP_