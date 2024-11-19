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

#include <onika/app/config_struct.h>

#include <onika/cuda/cuda.h>
#include <onika/plugin.h>
#include <vector>
#include <cstdint>
#include <string>
#include <map>

namespace onika
{

  namespace app
  {

    // associates a number of threads to a set of regular expressions selecting targeted operators' pathnames
    using StringVector = std::vector<std::string>;
    using StringIntMap = std::map<std::string,int>;
    using UInt64Vector = std::vector<uint64_t>;
    using IntVector = std::vector<long>;

    ONIKA_APP_CONFIG_Begin( logging                              , "log streams configuration" );
      ONIKA_APP_CONFIG_Item( bool        , parallel , false      , "Allows all processes (not only rank 0) to output to log streams" );
      ONIKA_APP_CONFIG_Item( bool        , debug    , false      , "Enable debug stream" );
      ONIKA_APP_CONFIG_Item( std::string , log_file , ""         , "write logs to file instead of standard output stream" );
      ONIKA_APP_CONFIG_Item( std::string , err_file , ""         , "write errors to file instead standard error stream");
      ONIKA_APP_CONFIG_Item( std::string , dbg_file , ""         , "write debug messages to file instead of standard output");
    ONIKA_APP_CONFIG_End();

    ONIKA_APP_CONFIG_Begin( trace                                  , "enables and configure execution traces");
      ONIKA_APP_CONFIG_Item( bool         , enable    , false      , "if true enables trace output with given parameters" );
      ONIKA_APP_CONFIG_Item( std::string  , format    , "yaml"     , "available formats are 'vite' and 'dot'" );
      ONIKA_APP_CONFIG_Item( std::string  , file      , "trace"         , "Enables VITE format trace output if non-empty. Value is file name to use" );
      ONIKA_APP_CONFIG_Item( std::string  , color     , "operator" , "task coloring scheme : 'operator', 'duration' or 'tag'" );
      ONIKA_APP_CONFIG_Item( bool         , total     , false      , "outputs total time per task in a separate file" );
      ONIKA_APP_CONFIG_Item( bool         , idle      , true       , "outputs idle amount plot into a separate file" );
      ONIKA_APP_CONFIG_Item( std::string  , trigger   , ""         , "starts output after the first occurence of this event" );
      ONIKA_APP_CONFIG_Item( IntVector    , trigger_interval,   {} , "filter trace betwwen two trigger markers with given indices" );
      ONIKA_APP_CONFIG_Item( long         , idle_resolution , 8192 , "sampling resolution for idle ratio plot" );
      ONIKA_APP_CONFIG_Item( long         , idle_smoothing  , 32   , "sampling smoothing for idle ratio plot" );
    ONIKA_APP_CONFIG_End();

    ONIKA_APP_CONFIG_Begin( profiling                                   , "enables and configure profiling and execution traces");
      ONIKA_APP_CONFIG_Item( bool         , resmem         , false      , "profile resident memory increase" );
      ONIKA_APP_CONFIG_Item( bool         , exectime       , false      , "Write execution time of each operator to standard output" );
      ONIKA_APP_CONFIG_Item( bool         , summary        , false      , "Prints a summuray of execution times at end of simulation" );
      ONIKA_APP_CONFIG_Item( StringVector , filter         , {}         , "if non empty, limits profiling to operators whose pathname matches one of the regex in list" );
      ONIKA_APP_CONFIG_Struct( trace );
    ONIKA_APP_CONFIG_End();

    ONIKA_APP_CONFIG_Begin( debug                                   , "debuggin and introspection features" );
      ONIKA_APP_CONFIG_Item( bool         , plugins        , false  , "print loaded plugins");
      ONIKA_APP_CONFIG_Item( bool         , config         , false  , "print configuration block values");
      ONIKA_APP_CONFIG_Item( bool         , yaml           , false  , "print flattened (after include resolution) yaml config");
      ONIKA_APP_CONFIG_Item( bool         , graph          , false  , "print operator graph");
      ONIKA_APP_CONFIG_Item( bool         , ompt           , false  , "ompt internal debug messages");  
      ONIKA_APP_CONFIG_Item( bool         , graph_addr     , false  , "print operators' addresses");
      ONIKA_APP_CONFIG_Item( int          , graph_lod      , 1      , "level of detail for operator graph");
      ONIKA_APP_CONFIG_Item( std::string  , graph_fmt      , "console" , "graph output format");
      ONIKA_APP_CONFIG_Item( bool         , graph_rsc      , false   , "output operator resource nodes");
      ONIKA_APP_CONFIG_Item( bool         , files          , false   , "print loaded input files");
      ONIKA_APP_CONFIG_Item( std::string  , rng            , ""      , "random number generator state dump file name");
      ONIKA_APP_CONFIG_Item( StringVector , graph_filter   , {}      , "list of regular experessions to select operators to exclude from graph output");
      ONIKA_APP_CONFIG_Item( StringVector , filter         , {}      , "list of regular expression to select operators taht have debug output enabled");
      ONIKA_APP_CONFIG_Item( UInt64Vector , particle       , {}      , "particle id list for particles to debug");
      ONIKA_APP_CONFIG_Item( bool         , particle_nbh   , false   , "enable particle neighborhood debugging");
      ONIKA_APP_CONFIG_Item( bool         , particle_ghost , false   , "enable ghost particles debugging");
      ONIKA_APP_CONFIG_Item( bool         , fpe            , false   , "enable floating point exceptions");
      ONIKA_APP_CONFIG_Item( int          , verbose        , 0       , "verbosity level");
      ONIKA_APP_CONFIG_Item( bool         , graph_exec     , false   , "component graph execution debug traces");
    ONIKA_APP_CONFIG_End();

    ONIKA_APP_CONFIG_Begin( onika                                                                    , "Onika sub-system configuration" );
      ONIKA_APP_CONFIG_Item( int          , parallel_task_core_mult , ONIKA_TASKS_PER_CORE           , "Number of OpenMP tasks per core when using task mode" );
      ONIKA_APP_CONFIG_Item( int          , parallel_task_core_add  , 0                              , "Additional number of OpenMP tasks when using task mode" );
      ONIKA_APP_CONFIG_Item( int          , gpu_sm_mult             , ONIKA_CU_MIN_BLOCKS_PER_SM     , "GPU number of blocks per SM" );
      ONIKA_APP_CONFIG_Item( int          , gpu_sm_add              , 0                              , "GPU number of blocks added to grid size" );
      ONIKA_APP_CONFIG_Item( int          , gpu_block_size          , ONIKA_CU_MAX_THREADS_PER_BLOCK , "GPU kernel block size" );
      ONIKA_APP_CONFIG_Item( StringVector , gpu_disable_filter      , {}                             , "list of regular expressions matching paths of operators whose access to the GPU is disabled" );
      ONIKA_APP_CONFIG_Item( StringVector , gpu_enable_filter       , {}                             , "list of regular expressions matching paths of operators with GPU enabled, regardless of gpu_disable_filter settings" );
    ONIKA_APP_CONFIG_End();

    ONIKA_APP_CONFIG_Begin( configuration                                , "Application configuration");  
      ONIKA_APP_CONFIG_Struct( logging );
      ONIKA_APP_CONFIG_Struct( profiling );
      ONIKA_APP_CONFIG_Struct( debug );
      ONIKA_APP_CONFIG_Struct( onika );

      ONIKA_APP_CONFIG_Item( bool          , nogpu               , false , "globally disables GPU usage, even if some are present, and prevent any call to Cuda or HIP libraries");
      ONIKA_APP_CONFIG_Item( bool          , mpimt               , true  , "enables MPI_THREAD_MULTIPLE feature if available");
      ONIKA_APP_CONFIG_Item( bool          , pinethreads         , false , "try to pine OpenMP thread");
      ONIKA_APP_CONFIG_Item( int           , threadrotate        , 0     , "rotate thread pinning (use OpenMP thread index + threadrotate modulus number of threads as a reference)");
      ONIKA_APP_CONFIG_Item( int           , omp_num_threads     , -1    , "number of OpenMP threads (-1 means default)");
      ONIKA_APP_CONFIG_Item( int           , omp_max_nesting     , -1    , "maximum nesting level for OpenMP nested parallelism");
      ONIKA_APP_CONFIG_Item( bool          , omp_nested          , false , "enables OpenMP nesting");
      ONIKA_APP_CONFIG_Item( StringIntMap  , omp_max_threads_filter      , {}                             , "list of regular expressions matching paths of operators with forbidden access to the GPU" );

      ONIKA_APP_CONFIG_Item( std::string   , plugin_dir          , onika::default_plugin_search_dir() , "plugin directory");  
      ONIKA_APP_CONFIG_Item( std::string   , plugin_db           , ""    , "plugin dictionary file");
      ONIKA_APP_CONFIG_Item( StringVector  , plugins             , {}    , "list of plugins forced to load");
      ONIKA_APP_CONFIG_Item( bool          , generate_plugins_db , false , "generate plugin data base and exit");

      ONIKA_APP_CONFIG_Item( std::string   , help                , ""    , "print help and exit");
      ONIKA_APP_CONFIG_Item( bool          , run_unit_tests      , false , "run unit tests and exit");  
      
      ONIKA_APP_CONFIG_Node( set , "override configuration items with those in this map" );
    ONIKA_APP_CONFIG_End();

    using ApplicationConfiguration = ONIKA_APP_CONFIG_Struct_configuration;

// end of namespaces
  }

}

