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

#pragma once

#include <onika/parallel/random.h>
#include <onika/plugin.h>

#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_set_from_regex.h>

#include <onika/string_utils.h>
#include <onika/yaml/yaml_utils.h>
#include <onika/file_utils.h>
#include <onika/log.h>
#include <onika/thread.h>
#include <onika/test/unit_test.h>
#include <onika/cpp_utils.h>

#include <onika/omp/version.h>
#include <onika/omp/ompt_interface.h>
#include <onika/parallel/parallel_execution_context.h>

#include <onika/yaml/cmdline.h>
#include <onika/app/default_app_config.h>
#include <onika/app/vite_profiler.h>
#include <onika/app/vite_operator_functions.h>
#include <onika/app/log_profiler.h>
#include <onika/app/debug_profiler.h>
#include <onika/app/dot_sim_graph.h>
		
#include <mpi.h>
#include <onika/mpi/mpi_parallel_stats.h>

#include <onika/trace/vite_trace_format.h>
#include <onika/trace/dot_trace_format.h>
#include <onika/trace/yaml_trace_format.h>

// dummy function to be used as a breakpoint marker just before simulation is ran
// usefull for adding breakpoints in loaded plugins
void simulation_start_breakpoint();
void plugins_loaded_breakpoint();

namespace onika
{
  namespace app
  {

    // ============== simple high level API ================

    struct ApplicationContext
    {
      std::shared_ptr<onika::app::ApplicationConfiguration> m_configuration;
      std::vector<std::string> m_input_files;
      YAML::Node m_cmdline_config;
      YAML::Node m_input_data;
      YAML::Node m_simulation_node;
      int m_cpucount = 1;
      int m_cpu_hw_threads = 1;
      std::vector<int> m_cpu_ids;
      int m_mpi_rank = 0;
      int m_mpi_nprocs = 1;
      int m_mpi_mt_support_level = MPI_THREAD_SINGLE;
      int m_mpi_external_init = 0;
      int m_ngpus = 0;
      int m_test_npassed = 0;
      int m_test_nfailed = 0;
      onika::trace::TraceOutputFormat * m_prof_trace = nullptr;
      bool m_need_profiling = false;
      const onika::PluginDBMap* m_plugin_db = nullptr;
      bool m_plugin_db_generate_mode = false;
      std::shared_ptr<onika::scg::OperatorNode> m_simulation_graph = nullptr;
      int m_return_code = -1;
      
      void set_multiple_run(bool yn);
      int get_error_code() const;
      onika::scg::OperatorNode* node(const std::string& nodepath) const;
    };
  
    std::shared_ptr<ApplicationContext>
    init(int argc, char const * const argv[]);

    void
    run(std::shared_ptr<ApplicationContext> ctx);

    void
    end(std::shared_ptr<ApplicationContext> ctx);




    // ============ detailed low-level application API ===========

    void
    initialize();

    void
    finalize( const onika::app::ApplicationConfiguration & configuration
                        , std::shared_ptr<onika::scg::OperatorNode> simulation_graph
                        , onika::trace::TraceOutputFormat * otf );

    std::pair< std::vector<std::string> , YAML::Node >
    parse_command_args( int argc, char const * const argv[] );

    std::tuple<YAML::Node,YAML::Node,onika::app::ApplicationConfiguration>
    load_yaml_input( const std::vector<std::string>& main_input_files, YAML::Node cmdline = YAML::Node(YAML::NodeType::Map) );

    onika::scg::OperatorNode*
    node_from_path( std::shared_ptr<onika::scg::OperatorNode> simulation_graph , const std::string& nodepath );

    std::tuple< int , int , std::vector<int> >
    intialize_openmp( onika::app::ApplicationConfiguration & configuration , int mpi_rank = 0 , bool allow_openmp_conf = true );

    std::string
    cpu_id_list_to_string(const std::vector<int>& cpu_ids);

    std::tuple<int,int,int,int>
    initialize_mpi( const onika::app::ApplicationConfiguration & configuration , int argc, char const * const argv[], MPI_Comm app_world_comm = MPI_COMM_WORLD );

    int
    initialize_gpu( const onika::app::ApplicationConfiguration & configuration );

    std::pair<int,int>
    run_embedded_tests( const onika::app::ApplicationConfiguration & configuration );

    std::pair< onika::trace::TraceOutputFormat * , bool >
    initialize_profiling( onika::app::ApplicationConfiguration & configuration, int mpi_rank, int nb_procs );

    std::pair< const onika::PluginDBMap* , bool >
    initialize_plugins( onika::app::ApplicationConfiguration & configuration );

    bool
    print_help( const onika::app::ApplicationConfiguration & configuration , const std::string& appname, const onika::PluginDBMap* plugin_db );

    std::shared_ptr<onika::scg::OperatorNode>
    build_simulation_graph( const onika::app::ApplicationConfiguration & configuration , YAML::Node simulation_node );

    void
    run_simulation_graph( std::shared_ptr<onika::scg::OperatorNode> simulation_graph , bool configuration_needs_profiling );

    template<class StreamT>
    inline StreamT& print_host_system_info(StreamT& out , const onika::app::ApplicationConfiguration & configuration
                                          , int nb_procs, int cpucount, int cpu_hw_threads, const std::vector<int>& cpu_ids, int n_gpus)
    {
      // host system info
      out << std::endl
           << "Version : "<< ONIKA_VERSION
#     ifndef NDEBUG
           <<" (debug)"
#     endif
           <<std::endl
           << "MPI     : "<< onika::format_string("%-4d",nb_procs)<<" process"<<onika::plurial_suffix(nb_procs,"es")<<std::endl
           << "CPU     : "<< onika::format_string("%-4d",cpucount)<<" core"<<onika::plurial_suffix(cpucount)
                          <<" (max "<<cpu_hw_threads<<") :"<<onika::app::cpu_id_list_to_string(cpu_ids)<<std::endl
           << "OpenMP  : "<< onika::format_string("%-4d",configuration.omp_num_threads) <<" thread"<<onika::plurial_suffix(configuration.omp_num_threads)
                          <<" (v"<< onika::omp::get_version_string() 
                          << ( ( configuration.omp_max_nesting > 1 ) ? onika::format_string(" nest=%d",configuration.omp_max_nesting) : std::string("") ) <<")"<<std::endl
           << "SIMD    : "<< onika::memory::simd_arch() << std::endl
           << "SOATL   : align="<<onika::memory::DEFAULT_ALIGNMENT<<" , vec="<<onika::memory::DEFAULT_CHUNK_SIZE << std::endl;
#     ifdef ONIKA_CUDA_VERSION
           out << ONIKA_CU_NAME_STR << "    : v"<< USTAMP_STR(ONIKA_CUDA_VERSION) ;
           if(n_gpus==0) out<< " (no GPU)";
           else if(n_gpus==1) out<< " (1 GPU)";
           else out<< " ("<< n_gpus<<" GPUs)";
           out << " , mem. align " << onika::memory::GenericHostAllocator::DefaultAlignBytes << std::endl;
#     endif
      return out;
    }

  // end of namespaces
  }
}

