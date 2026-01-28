#include "Decoder/Decoder_quaternary.hpp"
#include <vector>
#include <set>
#include <iostream>
#include <cmath>
using namespace std;

Decoder_quaternary::Decoder_quaternary(const int& K, const int& N, const int& L, const vector<bool>& frozen_bits) 
: Decoder(K, N), L(L), frozen_bits(frozen_bits)
{
	this->frozen_values = vector<int>(N, 0);
    this->active_paths.insert(0);
	int n = log2(N);
	int max_depth_llrs = n - 1;
    for (auto i = 0; i < L; i++) {
		auto new_tree = new Tree_metric<Contents_quaternary>(n, 0);
		this->polar_trees.push_back(new_tree);
		this->recursive_allocate_nodes_contents(new_tree->get_root(), this->N, max_depth_llrs);
		this->recursive_initialize_frozen_bits(new_tree->get_root(), frozen_bits);
    }
    for (auto i = 0; i < L; i++) 
        leaves_array.push_back(this->polar_trees[i]->get_leaves());
}

void Decoder_quaternary::set_frozen_values(const vector<int>& fv)
{
	this->frozen_values = fv;
}

void Decoder_quaternary::recursive_allocate_nodes_contents(Node<Contents_quaternary>* node_curr, const int vector_size, int& max_depth_llrs)
{
	node_curr->set_contents(new Contents_quaternary(vector_size));
	if (!node_curr->is_leaf())
	{
		const int new_vector_size = vector_size / 2;
		for (auto c : node_curr->get_children())
			this->recursive_allocate_nodes_contents(c, new_vector_size, max_depth_llrs);
	} else node_curr->get_c()->max_depth_llrs = max_depth_llrs;
	max_depth_llrs = this->polar_trees[0]->get_depth() - node_curr->get_depth();
	// the max_depth_llrs is the (path length -1) from the current leaf
	// to the common ancestor of itself and the previous immediate leaf.
	// max_depth_llrs for even leaf is always 0
}

void Decoder_quaternary::recursive_initialize_frozen_bits(const Node<Contents_quaternary>* node_curr, const std::vector<bool>& frozen_bits)
{
	if (!node_curr->is_leaf()) 
	{
		for (auto c : node_curr->get_children())
			this->recursive_initialize_frozen_bits(c, frozen_bits); 
	}
	else 
		node_curr->get_contents()->is_frozen_bit = frozen_bits[node_curr->get_lane_id()];
}

Decoder_quaternary::~Decoder_quaternary()
{
    for (auto i = 0; i < L; i++) 
        this->recursive_deallocate_nodes_contents(this->polar_trees[i]->get_root());
}

void Decoder_quaternary::recursive_deallocate_nodes_contents(Node<Contents_quaternary>* node_curr)
{
	for (auto c : node_curr->get_children())
		this->recursive_deallocate_nodes_contents(c); 

	delete node_curr->get_contents();
	node_curr->set_contents(nullptr);
}

void Decoder_quaternary::recursive_compute_llr(Node<Contents_quaternary>* node_cur, int depth)
{
	auto node_father = node_cur->get_father();

	if (depth != 0)
		recursive_compute_llr(node_father, --depth);

	if (!node_cur->is_root()) {
		int child_id = node_cur->get_child_id();
		auto p = node_cur->get_father();
		auto p_c = p->get_contents();
		int size = p_c->l.size(), size_half = size / 2;
		// p_fst for (u1+u2, v1+v2)
		// p_snd for (u2, v2)
		if (child_id == 1) {
			auto left_child = p->get_children()[0];
			// p_new for (u2, v2)
			f_minus_quaternary(p_c->l.data(), p_c->l.data() + size_half, left_child->get_contents()->s, 
				size_half, node_cur->get_contents()->l.data());
		} else if (child_id == 0) {
			// p_new for (u1, v1)
			f_plus_quaternary(p_c->l.data(), p_c->l.data() + size_half, size_half, node_cur->get_contents()->l.data());
		}
	}
}

void Decoder_quaternary::select_best_path()
{   // select the best one, not the best L ones.
	int best_path = 0;
	if (active_paths.size() >= 1)
		best_path = *active_paths.begin();

	for (int path : active_paths)
		if(polar_trees[path]->get_path_metric() < polar_trees[best_path]->get_path_metric())
			best_path = path;

	this->best_path = best_path;
}

void Decoder_quaternary::_load(double** Y_N)
{
	for (int path = 0; path < this->L; path++) {
		// TODO: this copy may not function correctly
		for (int i=0; i< this->N; i++)
			std::copy(Y_N[i], Y_N[i]+4, this->polar_trees[path]->get_root()->get_contents()->l[i].data());
		// this->polar_trees[path]->get_root()->get_contents()->l = Y_N;
		// polar_trees[path]->set_path_metric(numeric_limits<double>::min());
		polar_trees[path]->set_path_metric(0);
	}

	// initialization
	active_paths.clear();
	active_paths.insert(0);
}

void Decoder_quaternary::_decode()
{
	std::set<int> last_active_paths;
	int cur_path;
	int depth = log2(N);

	// tuples to be sorted. <Path, estimated bit (x,z), metric>
	std::vector<std::tuple<int,int,double>> metrics_vec;

	// run through each leaf
	for (auto leaf_index = 0; leaf_index < this->N; leaf_index++)
	{
		// compute LLR for current leaf
		for (auto path : active_paths) 
			this->recursive_compute_llr(leaves_array[path][leaf_index], 
										leaves_array[path][leaf_index]->get_c()->max_depth_llrs);
		// only need to compute llr starting from the common ancestor of itself and the previous leaf

		// if current leaf is a frozen bit
		if (leaves_array[0][leaf_index]->get_c()->is_frozen_bit) {
		    // penalize if the prediction for frozen bit is wrong
			int x = frozen_values[leaf_index]; // may not be 0
			auto min_phi = std::numeric_limits<double>::max();
			for (auto path : active_paths) {
				auto cur_leaf = leaves_array[path][leaf_index];
				cur_leaf->get_c()->s[0][0] = x;
				cur_leaf->get_c()->s[0][1] = x;
				auto phi_cur = phi_depolarize(polar_trees[path]->get_path_metric(), cur_leaf->get_c()->l[0], x, x);
				this->polar_trees[path]->set_path_metric(phi_cur);
				min_phi = std::min<double>(min_phi, phi_cur);
			}

			// normalization
			for (auto path : active_paths)
				this->polar_trees[path]->set_path_metric(this->polar_trees[path]->get_path_metric() - min_phi);

		} else {
			// metrics vec used to store values of hypothetic path metrics
			metrics_vec.clear();
			auto min_phi = std::numeric_limits<double>::max();
			for (auto path : active_paths) {
				auto cur_leaf = leaves_array[path][leaf_index];
				double phi0 = phi_depolarize(polar_trees[path]->get_path_metric(), cur_leaf->get_c()->l[0], 0, 0);
				double phi1 = phi_depolarize(polar_trees[path]->get_path_metric(), cur_leaf->get_c()->l[0], 0, 1);
				double phi2 = phi_depolarize(polar_trees[path]->get_path_metric(), cur_leaf->get_c()->l[0], 1, 0);
				double phi3 = phi_depolarize(polar_trees[path]->get_path_metric(), cur_leaf->get_c()->l[0], 1, 1);
				metrics_vec.push_back(std::make_tuple(path, 0, phi0));
				metrics_vec.push_back(std::make_tuple(path, 1, phi1));
				metrics_vec.push_back(std::make_tuple(path, 2, phi2));
				metrics_vec.push_back(std::make_tuple(path, 3, phi3));

				min_phi = std::min<double>(min_phi, phi0);
				min_phi = std::min<double>(min_phi, phi1);
				min_phi = std::min<double>(min_phi, phi2);
				min_phi = std::min<double>(min_phi, phi3);
			}

			
			for (auto& vec : metrics_vec) // normalization
				std::get<2>(vec) -= min_phi;
			if (active_paths.size() <= (unsigned)(L / 4)) {
				vector<int> decisions = {0,1,2,3};
				last_active_paths = active_paths;
				for (auto path : last_active_paths)
					this->duplicate_path(path, leaf_index, leaves_array, decisions);
			} else {
				// sort hypothetic path metrics in increasing order
				std::sort(metrics_vec.begin(), metrics_vec.end(),
					[](std::tuple<int,int,double> x, std::tuple<int,int,double> y){
						return std::get<2>(x) < std::get<2>(y);
					});

				// search in worst metrics. If a path is found four times, then remove it from activate paths
				for (auto it = metrics_vec.begin() + L; it != metrics_vec.end(); ++it) {
					cur_path = std::get<0>(*it);
					int cnt = 1;
					for (auto it_double = it+1; it_double != metrics_vec.end(); ++it_double) 
						if (std::get<0>(*it_double) == cur_path) cnt++;
					if (cnt == 4) active_paths.erase(std::get<0>(*it));
				}

				// remove worst metrics from list
				metrics_vec.resize(L);

				for (auto it = metrics_vec.begin(); it != metrics_vec.end(); ++it) {
					cur_path = std::get<0>(*it);
					vector<int> decisions;
					auto it_double = it + 1;
					while (it_double != metrics_vec.end()) {
						if (std::get<0>(*it_double) == cur_path) {
							decisions.push_back(std::get<1>(*it_double));
							it_double = metrics_vec.erase(it_double);
						} else ++it_double;
					}
					if (decisions.size() != 0) {
						decisions.push_back(std::get<1>(*it));
						duplicate_path(std::get<0>(*it), leaf_index, leaves_array, decisions); 
					} else { // choose
						int xz = std::get<1>(*it); // 2*u1 + v1
						int x = xz % 2, z = xz/2;
						leaves_array[std::get<0>(*it)][leaf_index]->get_c()->s[0][0] = x;
						leaves_array[std::get<0>(*it)][leaf_index]->get_c()->s[0][1] = z;
						polar_trees[std::get<0>(*it)]->set_path_metric(std::get<2>(*it));
					}
				}
			}
		}
		// right node keeps propagating sums, until itself becomes a left node
		for (auto path : active_paths)
			this->recursive_propagate_sums(leaves_array[path][leaf_index]);
	}

	this->select_best_path();
}

void Decoder_quaternary::recursive_propagate_sums(const Node<Contents_quaternary>* node_cur)
{
	auto children = node_cur->get_children();

	if (children.size() > 0) { // not leaf
		auto n_c = node_cur->get_contents();              // (u1+u2, v1+v2) || (u2, v2)
		int size = n_c->s.size(), size_half = size / 2;
		auto left_s = children[0]->get_contents()->s;     // (u1, v1)
		auto right_s = children[1]->get_contents()->s;    // (u2, v2)
		for (int i = 0; i < size_half; i++) {
			n_c->s[i][0] = left_s[i][0] ^ right_s[i][0];  // v1+v2
			n_c->s[i][1] = left_s[i][1] ^ right_s[i][1];  // u1+u2
		}
		for (int i = 0; i < size_half; i++) {
			n_c->s[i+size_half][0] = right_s[i][0];                 // v2
			n_c->s[i+size_half][1] = right_s[i][1];                 // u2
		}
	}
	if (!node_cur->is_root() && (node_cur->get_child_id() == 1)) // is the right child
		this->recursive_propagate_sums(node_cur->get_father());
	// else cerr << "recursive propagate sums ends at node at depth " << node_cur->get_depth() << " , lane_id " << node_cur->get_lane_id() << endl;
}

void Decoder_quaternary::duplicate_path(int path, int leaf_index, vector<vector<Node<Contents_quaternary>*>> leaves_array, vector<int>& decisions)
{
	vector<Node<Contents_quaternary>*> path_leaves, newpath_leaves;
	for (int i = 1; i < decisions.size(); i++) {
		int newpath = 0;
		while (active_paths.find(newpath++) != active_paths.end()){};
		newpath--;
		active_paths.insert(newpath);
		path_leaves = leaves_array[path];
		newpath_leaves = leaves_array[newpath];
		for (auto i = 0; i < leaf_index; i++)
			newpath_leaves[i]->get_c()->s = path_leaves[i]->get_c()->s;

		// the cleverer way
		recursive_duplicate_tree_sums(leaves_array[path][leaf_index], leaves_array[newpath][leaf_index], nullptr);
		if (leaf_index < this->N - 1)
			recursive_duplicate_tree_llr(leaves_array[path][leaf_index + 1], leaves_array[newpath][leaf_index + 1]);
		// do not need to copy the whole tree, only copy the necessary part to compute llr for all the future leaf nodes

		int xz = decisions[i];
		int x = xz%2, z = xz/2;
		leaves_array[newpath][leaf_index]->get_c()->s[0][0] = x;
		leaves_array[newpath][leaf_index]->get_c()->s[0][1] = z;
		polar_trees[newpath]->set_path_metric(phi_depolarize(polar_trees[path]->get_path_metric(),
															leaves_array[path][leaf_index]->get_c()->l[0], z, x));
	}
	int xz = decisions[0];
	int x = xz%2, z = xz/2;
	leaves_array[path][leaf_index]->get_c()->s[0][0] = x;
	leaves_array[path][leaf_index]->get_c()->s[0][1] = z;
	polar_trees[path]->set_path_metric(phi_depolarize(polar_trees[path]->get_path_metric(),
	                                                  leaves_array[path][leaf_index]->get_c()->l[0], z, x));
}

void Decoder_quaternary::recursive_duplicate_tree_llr(Node<Contents_quaternary>* node_a, Node<Contents_quaternary>* node_b)
{
	node_b->get_c()->l = node_a->get_c()->l;

	if (!node_a->get_father()->is_root())
		this->recursive_duplicate_tree_llr(node_a->get_father(), node_b->get_father());
}

void Decoder_quaternary::recursive_duplicate_tree_sums(Node<Contents_quaternary>* node_a, Node<Contents_quaternary>* node_b, Node<Contents_quaternary>* node_caller)
{
	if (!node_a->is_leaf()) { // if called by its right child
		auto left_child = (node_a->get_children())[0];
		if (left_child != node_caller) {
			node_b->get_children()[0]->get_c()->s = left_child->get_c()->s;
		}
	}
	if (!node_a->is_root())
		this->recursive_duplicate_tree_sums(node_a->get_father(), node_b->get_father(), node_a);
}

void Decoder_quaternary::_store(int** V) const
{
    auto *root = this->polar_trees[this->best_path]->get_root();
	for (int i=0; i<this->N; i++) {
		V[i][0] = root->get_c()->s[i][0];
		V[i][1] = root->get_c()->s[i][1];
	}
	// V = root->get_c()->s;
}

int Decoder_quaternary::decode(double** Y_N, int** V_K)
{
    this->_load(Y_N);
    this->_decode();
    this->_store(V_K);
	return 0;
}

void Decoder_quaternary::decode_SC(double** Y_N, int** V_K) 
{
	this->_load(Y_N);
	this->_decode_SC(polar_trees[0]->get_root());
	this->_store(V_K);
}

void Decoder_quaternary::_decode_SC(Node<Contents_quaternary>* node_cur) 
{
	auto n_c = node_cur->get_c();
	if (node_cur->is_leaf()) {
		if (n_c->is_frozen_bit) {
			n_c->s[0][0] = 0; n_c->s[0][1] = 0;
			return;
		}
		auto& p = n_c->l[0];
		auto& s = n_c->s[0];
		int max_id = 0;
		double temp_max = p[0];
		for (int i = 1; i < 4; i++) {
			if (p[i] > temp_max) {
				max_id = i;
				temp_max = p[i];
			}
		}
		switch(max_id) {
			case 0:
				s[0] = 0; s[1] = 0; break;	 // I = (0,0)
			case 1:
				s[0] = 1; s[1] = 0; break;   // X = (0,1)
			case 2:
				s[0] = 0; s[1] = 1; break;   // Z = (1,0)
			case 3:
				s[0] = 1; s[1] = 1; break;   // Y = (1,1)
		}
		return;
	}
	auto left_child = node_cur->get_children()[0];
	auto right_child = node_cur->get_children()[1];
	int size = n_c->s.size(), size_half = size / 2;
	f_plus_quaternary(n_c->l.data(), n_c->l.data() + size_half, size_half, left_child->get_c()->l.data());
	_decode_SC(left_child);
	f_minus_quaternary(n_c->l.data(), n_c->l.data() + size_half, left_child->get_c()->s, size_half, right_child->get_c()->l.data());
	_decode_SC(right_child);
	auto left_s = left_child->get_c()->s;
	auto right_s = right_child->get_c()->s;
	for (int i = 0; i < size_half; i++) {
		n_c->s[i][0] = left_s[i][0] ^ right_s[i][0];  // v1 + v2
		n_c->s[i][1] = left_s[i][1] ^ right_s[i][1];  // u1 + u2
	}
	for (int i = 0; i < size_half; i++) {
		n_c->s[i+size_half][0] = right_s[i][0];       // v2
		n_c->s[i+size_half][1] = right_s[i][1];       // u2
	}
}