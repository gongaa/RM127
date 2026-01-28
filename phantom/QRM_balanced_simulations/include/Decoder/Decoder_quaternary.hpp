#ifndef DECODER_QUATERNARY_HPP_
#define DECODER_QUATERNARY_HPP_
#include <vector>
#include <set>
#include "Decoder.hpp"
#include "Tree.hpp"
#include "Util/Util.hpp"
using namespace std;

class Contents_quaternary
{
public:
    vector<vector<double>> l;     // probability (TODO: make it LLR) array (p_I, p_X, p_Z, p_Y)
    vector<vector<int>> s;       // partial sum array (u_x, u_z)

    bool is_frozen_bit;
    int max_depth_llrs;

    explicit Contents_quaternary(int size) : l(size, vector<double>(4,0)), s(size, vector<int>(2,0)), is_frozen_bit(false) {}
    virtual ~Contents_quaternary() {}
};

class Decoder_quaternary : Decoder
{
protected:
    const int L;             // maximum path number
    std::set<int> active_paths;
    int best_path;
    vector<vector<int>> vs;  // for precoded polar codes

    vector<bool> frozen_bits;
    vector<int> frozen_values; // support arbitrary frozen values (not all-zero)
    vector<Tree_metric<Contents_quaternary>*> polar_trees;
    vector<vector<Node<Contents_quaternary>*>> leaves_array;   

public:
    Decoder_quaternary(const int& K, const int& N, const int& L, const vector<bool>& frozen_bits);
    virtual ~Decoder_quaternary();
    virtual int decode(double** Y_N, int** V_K);
    void decode_SC(double** Y_N, int** V_K);
    void set_frozen_values(const vector<int>& fv);

protected:
    void _load(double** Y_N);
    void _decode();
    void _store(int** V_K) const;
    void _decode_SC(Node<Contents_quaternary>* node_cur);

private:
    void recursive_compute_llr(Node<Contents_quaternary>* node_cur, int depth);
    void recursive_propagate_sums(const Node<Contents_quaternary>* node_cur);
    void duplicate_path(int path, int leaf_index, vector<vector<Node<Contents_quaternary>*>> leaves_array, vector<int>& decisions);

    void recursive_duplicate_tree_llr(Node<Contents_quaternary>* node_a, Node<Contents_quaternary>* node_b);
    void recursive_duplicate_tree_sums(Node<Contents_quaternary>* node_a, Node<Contents_quaternary>* node_b, Node<Contents_quaternary>* node_caller);

protected:
    virtual void select_best_path();
    
    void recursive_allocate_nodes_contents(Node<Contents_quaternary>* node_curr, const int vector_size, int& max_depth_llrs);
    void recursive_initialize_frozen_bits(const Node<Contents_quaternary>* node_curr, const std::vector<bool>& frozen_bits);
    void recursive_deallocate_nodes_contents(Node<Contents_quaternary>* node_curr);

};

#endif /* DECODER_QUATERNARY_HPP_ */