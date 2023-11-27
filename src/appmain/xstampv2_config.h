#pragma once

#include "yaml_config_struct.h"

#include <onika/cuda/cuda.h>
#include <vector>
#include <cstdint>
#include <string>
#include <map>

// associates a number of threads to a set of regular expressions selecting targeted operators' pathnames
using StringVector = std::vector<std::string>;
using UInt64Vector = std::vector<uint64_t>;
using IntVector = std::vector<long>;

xsv2ConfigBegin( logging                              , "log streams configuration" );
  xsv2ConfigItem( bool        , parallel , false      , "Allows all processes (not only rank 0) to output to log streams" );
  xsv2ConfigItem( bool        , debug    , false      , "Enable debug stream" );
  xsv2ConfigItem( std::string , log_file , ""         , "write logs to file instead of standard output stream" );
  xsv2ConfigItem( std::string , err_file , ""         , "write errors to file instead standard error stream");
  xsv2ConfigItem( std::string , dbg_file , ""         , "write debug messages to file instead of standard output");
xsv2ConfigEnd();

xsv2ConfigBegin( trace                                  , "enables and configure execution traces");
  xsv2ConfigItem( bool         , enable    , false      , "if true enables trace output with given parameters" );
  xsv2ConfigItem( std::string  , format    , "yaml"     , "available formats are 'vite' and 'dot'" );
  xsv2ConfigItem( std::string  , file      , "trace"         , "Enables VITE format trace output if non-empty. Value is file name to use" );
  xsv2ConfigItem( std::string  , color     , "operator" , "task coloring scheme : 'operator', 'duration' or 'tag'" );
  xsv2ConfigItem( bool         , total     , false      , "outputs total time per task in a separate file" );
  xsv2ConfigItem( bool         , idle      , true       , "outputs idle amount plot into a separate file" );
  xsv2ConfigItem( std::string  , trigger   , ""         , "starts output after the first occurence of this event" );
  xsv2ConfigItem( IntVector    , trigger_interval,   {} , "filter trace betwwen two trigger markers with given indices" );
  xsv2ConfigItem( long         , idle_resolution , 8192 , "sampling resolution for idle ratio plot" );
  xsv2ConfigItem( long         , idle_smoothing  , 32   , "sampling smoothing for idle ratio plot" );
xsv2ConfigEnd();

xsv2ConfigBegin( profiling                                   , "enables and configure profiling and execution traces");
  xsv2ConfigItem( bool         , resmem         , false      , "profile resident memory increase" );
  xsv2ConfigItem( bool         , exectime       , false      , "Write execution time of each operator to standard output" );
  xsv2ConfigItem( bool         , summary        , false      , "Prints a summuray of execution times at end of simulation" );
  xsv2ConfigItem( StringVector , filter         , {}         , "if non empty, limits profiling to operators whose pathname matches one of the regex in list" );
  xsv2ConfigStruct( trace );
xsv2ConfigEnd();

xsv2ConfigBegin( debug                                   , "debuggin and introspection features" );
  xsv2ConfigItem( bool         , plugins        , false  , "print loaded plugins");
  xsv2ConfigItem( bool         , config         , false  , "print configuration block values");
  xsv2ConfigItem( bool         , yaml           , false  , "print flattened (after include resolution) yaml config");
  xsv2ConfigItem( bool         , graph          , false  , "print operator graph");
  xsv2ConfigItem( bool         , ompt           , false  , "ompt internal debug messages");  
  xsv2ConfigItem( bool         , graph_addr     , false  , "print operators' addresses");
  xsv2ConfigItem( int          , graph_lod      , 1      , "level of detail for operator graph");
  xsv2ConfigItem( std::string  , graph_fmt      , "console" , "graph output format");
  xsv2ConfigItem( bool         , graph_rsc      , false   , "output operator resource nodes");
  xsv2ConfigItem( bool         , files          , false   , "print loaded input files");
  xsv2ConfigItem( std::string  , rng            , ""      , "random number generator state dump file name");
  xsv2ConfigItem( StringVector , graph_filter   , {}      , "list of regular experessions to select operators to exclude from graph output");
  xsv2ConfigItem( StringVector , filter         , {}      , "list of regular expression to select operators taht have debug output enabled");
  xsv2ConfigItem( UInt64Vector , particle       , {}      , "particle id list for particles to debug");
  xsv2ConfigItem( bool         , particle_nbh   , false   , "enable particle neighborhood debugging");
  xsv2ConfigItem( bool         , particle_ghost , false   , "enable ghost particles debugging");
  xsv2ConfigItem( bool         , fpe            , false   , "enable floating point exceptions");
  xsv2ConfigItem( int          , verbose        , 0       , "verbosity level");
  xsv2ConfigItem( bool         , graph_exec     , false   , "component graph execution debug traces");
xsv2ConfigEnd();

xsv2ConfigBegin( onika                                                                    , "Onika sub-system configuration" );
  xsv2ConfigItem( int          , parallel_task_core_mult , ONIKA_TASKS_PER_CORE           , "Number of OpenMP tasks per core when using task mode" );
  xsv2ConfigItem( int          , parallel_task_core_add  , 0                              , "Additional number of OpenMP tasks when using task mode" );
  xsv2ConfigItem( int          , gpu_sm_mult             , ONIKA_CU_MIN_BLOCKS_PER_SM     , "GPU number of blocks per SM" );
  xsv2ConfigItem( int          , gpu_sm_add              , 0                              , "GPU number of blocks added to grid size" );
  xsv2ConfigItem( int          , gpu_block_size          , ONIKA_CU_MAX_THREADS_PER_BLOCK , "GPU kernel block size" );
  xsv2ConfigItem( StringVector , gpu_disable_filter      , {}                             , "list of regular expressions matching paths of operators with forbidden access to the GPU" );
xsv2ConfigEnd();

xsv2ConfigBegin( configuration                                , "exaStampV2 configuration");  
  xsv2ConfigStruct( logging );
  xsv2ConfigStruct( profiling );
  xsv2ConfigStruct( debug );
  xsv2ConfigStruct( onika );

  xsv2ConfigItem( bool          , mpimt               , true  , "enables MPI_THREAD_MULTIPLE feature if available");
  xsv2ConfigItem( bool          , pinethreads         , false , "try to pine OpenMP thread");
  xsv2ConfigItem( int           , omp_num_threads     , -1    , "number of OpenMP threads (-1 means default)");
  xsv2ConfigItem( int           , omp_max_nesting     , -1    , "maximum nesting level for OpenMP nested parallelism");
  xsv2ConfigItem( bool          , omp_nested          , false , "enables OpenMP nesting");

  xsv2ConfigItem( std::string   , plugin_dir          , USTAMP_PLUGIN_DIR , "plugin directory");  
  xsv2ConfigItem( std::string   , plugin_db           , ""    , "plugin dictionary file");
  xsv2ConfigItem( StringVector  , plugins             , {}    , "list of plugins forced to load");
  xsv2ConfigItem( bool          , generate_plugins_db , false , "generate plugin data base and exit");

  xsv2ConfigItem( std::string   , help                , ""    , "print help and exit");
  xsv2ConfigItem( bool          , run_unit_tests      , false , "run unit tests and exit");  
  
  xsv2ConfigNode( set , "override configuration items with those in this map" );
xsv2ConfigEnd();

