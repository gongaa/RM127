#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "Encoder/Encoder_RM.hpp"
#include "Simulation/Simulation.hpp"
#include "Channel/Channel.hpp"

using namespace std;

static void show_usage(string name)
{
  cerr << "Usage: " << name << " <options> SOURCES"
       << "Options:\n\t"
       << left << setw(20) << "-h,--help" << "Show this help message\n\t"
       << left << setw(20) << "-rz" << "Specify the order of stabilizers\n\t"
       << left << setw(20) << "-px" << "Specify the bit-flip error rate\n\t"
       << left << setw(20) << "-n,--num_samples" << "Specify the number of samples to perform simulation on\n\t"
       << left << setw(20) << "-l,--list_size" << "Specify the list size used in the SCL decoder\n\t"
       << left << setw(20) << "-seed" << "Specify the random seed\n\t"
       << left << setw(20) << "-interval" << "Specify the printing interval, default is 1000\n\t"
       << endl;
}

int main(int argc, char** argv)
{
  int m, rx, rz; // for Reed-Muller
  int list_size = 8, n = 1000;
  double px; 
  int seed = 42;
  int print_interval = 1000;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      show_usage(argv[0]);
      return 0;
    } 
    std::istringstream iss( argv[++i] );
    if (arg == "-m") {
      iss >> m;
    } else if (arg == "-rx") {
      iss >> rx;
    } else if (arg == "-rz") {
      iss >> rz;
    } else if (arg == "-px") {
      iss >> px;  
    } else if ((arg == "-n") || (arg == "--num_samples")) {
      iss >> n;
    } else if ((arg == "-l") || (arg == "--list_size")) {
      iss >> list_size;
    } else if (arg == "-seed") {
      iss >> seed;
    } else if (arg == "-interval") {
      iss >> print_interval;
    } else {
      cerr << "Argument not supported." << endl;
      return 1;
    }
  }

  auto start = chrono::high_resolution_clock::now();

  simulation_punctured_RM(rz, n, px, list_size, seed);

  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
  cerr << "Finish in " << duration.count() << " s" << endl;

  return 0;
}