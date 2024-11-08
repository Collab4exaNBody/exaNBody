/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exanb/core/parallel_random.h>
#include <onika/thread.h>
#include <mpi.h>
#include <yaml-cpp/yaml.h>
#include <omp.h>
#include <sstream>
#include <unordered_map>

namespace exanb
{

  namespace rand
  {
    static std::random_device g_random_device;
    static std::unordered_map< size_t, std::mt19937_64 > g_thread_random_engine;

    std::mt19937_64 & random_engine()
    {
      return g_thread_random_engine[ onika::get_thread_index() ];
    }

    void generate_seed()
    {
#     pragma omp parallel
      {
        size_t tidx = onika::get_thread_index();
#       pragma omp critical(generate_seed_cs)
        {        
          g_thread_random_engine[ tidx ].seed( g_random_device() );
        }
      }
    }

    void set_seed(uint64_t seed)
    {
      int np = 1;
      int rank = 0;
      int initialized = 0;
      MPI_Initialized( &initialized );
      if( initialized )
      {
        MPI_Comm_size(MPI_COMM_WORLD , &np);
        MPI_Comm_rank(MPI_COMM_WORLD , &rank);
      }
#     pragma omp parallel
      {
        std::mt19937_64 r( seed + rank * omp_get_num_threads() + omp_get_thread_num() );
        r(); r();
        size_t tidx = onika::get_thread_index();
#       pragma omp critical(set_seed_cs)
        {
          g_thread_random_engine[ tidx ].seed( r() );
        }
      }
    }

    YAML::Node save_state()
    {
      int np = 1;
      int rank = 0;
      int initialized = 0;
      MPI_Initialized( &initialized );
      if( initialized )
      {
        MPI_Comm_size(MPI_COMM_WORLD , &np);
        MPI_Comm_rank(MPI_COMM_WORLD , &rank);
      }
      YAML::Node config;
      YAML::Node mpi_node;
      mpi_node["rank"] = rank;
      
#     pragma omp critical(save_state_cs)
      for(const auto &p : g_thread_random_engine)
      {
        YAML::Node thread_node;
        std::ostringstream oss;
        oss << p.second;
        thread_node["thread"] = p.first;
        thread_node["seed"] = oss.str();
        mpi_node["state"].push_back( thread_node );
      }

      config.push_back( mpi_node );
      return config;
    }

    void load_state(const YAML::Node& config)
    {
      int np = 1;
      int rank = 0;
      int initialized = 0;
      MPI_Initialized( &initialized );
      if( initialized )
      {
        MPI_Comm_size(MPI_COMM_WORLD , &np);
        MPI_Comm_rank(MPI_COMM_WORLD , &rank);
      }
      generate_seed(); // if seeds are missing from config file, new ones are generated
      
      YAML::Node mpi_node;
      for (std::size_t i=0;i<config.size();i++)
      {
        YAML::Node node = config[i];
        if( node["rank"].as<int>() == rank ) { mpi_node = node; }
      }
      std::map<int,std::string> thread_seeds;
      for(auto node : mpi_node["state"])
      {
        thread_seeds[ node["thread"].as<int>() ] = node["seed"].as<std::string>();
      }
#     pragma omp parallel
      {
        int tidx = onika::get_thread_index();
        if( thread_seeds.find(tidx) != thread_seeds.end() )
        {
          std::istringstream iss( thread_seeds[tidx] );
          iss >> g_thread_random_engine[ tidx ];
        }
      }
    }

  } // end of namespace rand

}


