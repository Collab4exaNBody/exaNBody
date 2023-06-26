#pragma once

#include <random>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace exanb
{
  namespace rand
  {
    std::mt19937_64& random_engine();

    void generate_seed();
    void set_seed(uint64_t seed);

    YAML::Node save_state();
    void load_state(const YAML::Node& config);
  }
}


