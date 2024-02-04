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
#include <exanb/core/plugin.h>

#include <exanb/core/operator_factory.h>
//#include "exanb/potential/pair_potential_factory.h"

#include <exanb/core/string_utils.h>
#include <exanb/core/yaml_utils.h>
#include <exanb/core/file_utils.h>
#include <exanb/core/log.h>
#include <exanb/core/grid.h>
#include <exanb/core/thread.h>
#include <exanb/core/unit_test.h>
#include <exanb/core/cpp_utils.h>

//#include "exanb/debug/debug_particle_id.h"

#include <onika/omp/ompt_interface.h>
#include <onika/parallel/parallel_execution_context.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <thread>
#include <cfenv>
#include <cmath>

#include <mpi.h>
#include <exanb/mpi/mpi_parallel_stats.h>

#include <yaml-cpp/yaml.h>

#include "cmdline.h"
#include "debug_profiler.h"
#include "log_profiler.h"
#include "vite_profiler.h"
#include <onika/trace/vite_trace_format.h>
#include <onika/trace/dot_trace_format.h>
#include <onika/trace/yaml_trace_format.h>
#include "vite_operator_functions.h"
#include "dot_sim_graph.h"
#include "operator_set_from_regex.h"

#include "xstampv2_config.h"

double xstamp_get_omp_version()
{
# ifndef XSTAMP_OMP_VERSION
  double version = 1.0;
  std::pair<unsigned long,double> version_dates [] = { {200505,2.5},{200805,3.0},{201107,3.1},{201307,4.0},{201511,4.5},{201811,5.0},{202011,5.1} };
  for(int i=0;i<7;i++)
  {
    if( version_dates[i].first < _OPENMP ) version = version_dates[i].second;
  }
  return version;
# else
  return XSTAMP_OMP_VERSION;
# endif
}
std::string xstamp_get_omp_version_string()
{
  double v = xstamp_get_omp_version();
  std::ostringstream oss;
  oss<< static_cast<int>(std::floor(v)) << '.' << ( static_cast<int>(std::floor(v*10))%10 );
  return oss.str();
}

std::string xstamp_grid_variants_as_string()
{
# define _XSTAMP_FIELD_SET_AS_STRING(FS) " " #FS
  return XSTAMP_FOR_EACH_FIELD_SET(_XSTAMP_FIELD_SET_AS_STRING);
# undef _XSTAMP_FIELD_SET_AS_STRING
}

// dummy function to be used as a breakpoint marker just before simulation is ran
// usefull for adding breakpoints in loaded plugins
void simulation_start_breakpoint() {}
void plugins_loaded_breakpoint() {}

int main(int argc,char*argv[])
{
  using namespace exanb;
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::string;
  using std::vector;

# ifndef NDEBUG
  std::cout << "to debug, use 'b simulation_start_breakpoint()' in gdb to stop program when all symbols are loaded"<<std::endl;
# endif

  if( argc < 2 )
  {
    lerr<<"Usage: "<<argv[0]<<" <input-file> [opt1,opt2...]"<<endl;
    lerr<<"   Or: "<<argv[0]<<" --help [operator-name]"<<endl;
    return 1;
  }

  int start_opt_arg = 1;
  std::vector<std::string> main_input_files;
  while(start_opt_arg < argc && std::string(argv[start_opt_arg]).find("--")!=0 )
  {
    main_input_files.push_back( argv[start_opt_arg] );
    ++ start_opt_arg;
  }

  // ======== read YAML input files and command line ===========

  // load user file and all subsequent include includes.
  // when no includes is specified, USTAMP_DEFAULT_CONFIG_FILE is loaded as if it has been included.
  // to prevent any file from being included, write "includes: []" in your input file
  vector<string> files_to_load = resolve_config_file_includes( argv[0] , main_input_files );
  assert( ! files_to_load.empty() );
  
  // merge YAML nodes from inner most included files up to user provided file
  YAML::Node input_data(YAML::NodeType::Map);
  for(auto f:files_to_load)
  {
    string pf = config_file_path(dirname(argv[0]),f);
    ldbg << "load config file "<< pf << std::endl; lout << std::flush;
    input_data = merge_nodes( YAML::Clone(input_data) , yaml_load_file_abort_on_except(pf) );
  }
  
  // additional arguments are interpreted as YAML strings that are parsed, and merged on top of files read previously
  command_line_options_to_yaml_config(argc,argv,start_opt_arg,input_data);
  // ======================================================


  // ====== extract YAML information blocks =========
  YAML::Node config_node;
  if( input_data["configuration"] )
  {
    config_node = input_data["configuration"];
    input_data = remove_map_key( input_data, "configuration" );
  }

  // convert YAML configuration node to data structure
  xsv2ConfigStruct_configuration configuration { config_node };

  // allow special block configuration block "set" to overload base input data
  if( configuration.set.IsMap() && configuration.set.size()>0 )
  {
    input_data = merge_nodes( input_data , configuration.set );
    configuration.set = YAML::Node();
    config_node = remove_map_key( config_node, "set" );
  }
  
  // simulation definition
  YAML::Node simulation_node;
  if( input_data["simulation"] )
  {
    simulation_node = input_data["simulation"];
    input_data = remove_map_key( input_data, "simulation" );
  }
  
  // random number generator state
  YAML::Node rng_node(YAML::NodeType::Null);
  if( input_data["random_generator_state"] )
  {
    rng_node = input_data["random_generator_state"];
    input_data = remove_map_key( input_data, "random_generator_state" );
  }
  // ===================================
  

  // ======== process debugging options =============
  OperatorNodeFactory::set_debug_verbose_level( configuration.debug.verbose );
  if( configuration.debug.fpe ) { feenableexcept( FE_ALL_EXCEPT & ~FE_INEXACT ); }
  // ===============================================


  // ============= OpenMP Initialization =============
  if( configuration.omp_num_threads > 0 )
  {
    ldbg << "configuration.omp_num_threads = "<<configuration.omp_num_threads<<std::endl;
    std::string num_thread_str;
    const char* old_num_thread_str = std::getenv("OMP_NUM_THREADS");
    if( old_num_thread_str != nullptr ) num_thread_str = old_num_thread_str;
    if( ! num_thread_str.empty() )
    {
      std::cout << "*WARNING* Overriding OpenMP's OMP_NUM_THREADS variable from "<<num_thread_str<<" to "<<configuration.omp_num_threads<<" because user speicified a positive value for configuration.omp_num_threads"<<std::endl;
    }    
    std::ostringstream oss;
    oss << configuration.omp_num_threads;
    num_thread_str = oss.str();
    setenv( "OMP_NUM_THREADS" , num_thread_str.c_str() , 1 );
    omp_set_num_threads(configuration.omp_num_threads);
  }
  
  if( configuration.omp_max_nesting > 1 )
  {
    if(configuration.omp_nested) { omp_set_nested(1); }
    omp_set_max_active_levels( configuration.omp_max_nesting );
  }
  int num_threads = 0;
# pragma omp parallel
  {
#   pragma omp single
    {
      num_threads = omp_get_num_threads();
    }
  }
  // ===============================================


  // ============= CPU Core counting and thread pining =============
  int cpucount = 0;
  std::vector<int> cpu_ids;
  {
    cpu_set_t cpuaff;
    CPU_ZERO(&cpuaff);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff);

    /* try to get as much threads as possible to get user limit */
    cpu_set_t cpuaff_backup = cpuaff;
    for (int i=0; i<CPU_SETSIZE; ++i)
    {
      CPU_SET(i, &cpuaff);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff);
    /***************/

    int ncpus = CPU_COUNT(&cpuaff);

    /* restore initial mask */    
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff_backup);
    /************************/
    
    for (int i=0; i<CPU_SETSIZE; ++i)
    {
      if (CPU_ISSET(i, &cpuaff)) 
      {
        cpu_ids.push_back(i);
      }
    }
    cpucount = cpu_ids.size();
    if( cpucount != ncpus )
    {
      lerr << "Internal error when counting CPU cores" << std::endl;
      std::abort();
    }
    if( cpucount<num_threads && configuration.pinethreads )
    {
      lerr << "Thread pining disabled because there are less cpu cores ("<<cpucount<<") than OpenMP threads ("<<num_threads<<")" << std::endl;
      configuration.pinethreads = false;
    }
  }
  // get the real core count (not the cpumask)
  int cpu_hw_threads = std::thread::hardware_concurrency();

  // ============= MPI Initialization =============
  int rank=0, nb_procs=0;
  int support = 0;
  if( configuration.mpimt )
  {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &support);
  }
  else
  {
    MPI_Init(&argc, &argv);
  }
  // scoped variable that properly finalizes MPI upon main function exit
  struct MpiScopedFinalizer { ~MpiScopedFinalizer() { MPI_Finalize(); } } mpi_finalize_on_scope_exit;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
  if( configuration.mpimt && support != MPI_THREAD_MULTIPLE && rank==0 )
  {
    lerr<<"Warning: no MPI_THREAD_MULTIPLE support. Concurrent calls to mpi API will fail."<<std::endl;
  }
  // ===============================================


  // =========== optional thread pinning ===========
  if( configuration.pinethreads )
  {
#   pragma omp parallel
    {
      cpu_set_t thread_cpu_aff;
      CPU_ZERO(&thread_cpu_aff);
      int tid = omp_get_thread_num();
      CPU_SET(cpu_ids[ ( tid + rank * configuration.threadrotate ) % cpu_ids.size() ], &thread_cpu_aff);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &thread_cpu_aff);
    }
  }
  // ===============================================


  // =========== configure logging system ===========
  configure_logging(
    configuration.logging.debug,
    configuration.logging.parallel,
    configuration.logging.log_file,
    configuration.logging.err_file,
    configuration.logging.dbg_file,
    rank,nb_procs);
  // ===============================================

  // get number of available GPUs, if any   
  int n_gpus = 0;
# ifdef XNB_CUDA_VERSION
  onika::cuda::CudaContext::set_global_gpu_enable( ! configuration.nogpu );
  if( onika::cuda::CudaContext::global_gpu_enable() )
  {
    ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_DEVICE_COUNT(&n_gpus) );
  }
# else
  onika::cuda::CudaContext::set_global_gpu_enable( false );
  n_gpus = 0;
# endif
  onika::memory::GenericHostAllocator::set_cuda_enabled( n_gpus > 0 );

  // generate a compact string representing cpu set assigned to current process
  std::string core_config;
  {
    std::ostringstream oss;
    int cs=-1, ce=-1;
    for(auto c:cpu_ids)
    {
      if(cs==-1) cs=ce=c;
      else
      {
        if(c==(ce+1)) ++ce;
        else
        {
          if(cs>=0) oss<<" "<<cs;
          if(ce>cs) oss<<"-"<<ce;
          cs=ce=c;
        }
      }
    }
    if(cs>=0) oss<<" "<<cs;
    if(ce>cs) oss<<"-"<<ce;
    core_config = oss.str();
  }

  // host system info
  lout << endl
       << "Version : "<< USTAMP_VERSION
# ifndef NDEBUG
       <<" (debug)"
# endif
       <<endl
       << "MPI     : "<< format_string("%-4d",nb_procs)<<" process"<<plurial_suffix(nb_procs,"es")<<endl
       << "CPU     : "<< format_string("%-4d",cpucount)<<" core"<<plurial_suffix(cpucount)<<" (max "<<cpu_hw_threads<<") :"<<core_config<<std::endl
       << "OpenMP  : "<< format_string("%-4d",num_threads) <<" thread"<<plurial_suffix(num_threads) <<" (v"<< xstamp_get_omp_version_string() 
                      << ( ( configuration.omp_max_nesting > 1 ) ? format_string(" nest=%d",configuration.omp_max_nesting) : std::string("") ) <<")"<<endl
       << "SIMD    : "<< onika::memory::simd_arch() << endl
       << "SOATL   : HFA P"<<XSTAMP_FIELD_ARRAYS_STORE_COUNT<<" / A"<<onika::memory::DEFAULT_ALIGNMENT<<" / C"<<onika::memory::DEFAULT_CHUNK_SIZE << endl;
# ifdef XNB_CUDA_VERSION
       lout << ONIKA_CU_NAME_STR << "    : v"<< USTAMP_STR(XNB_CUDA_VERSION);
       if(n_gpus==0) lout<< " (no GPU)"<< endl;
       else if(n_gpus==1) lout<< " (1 GPU)"<< endl;
       else lout<< " ("<< n_gpus<<" GPUs)"<< endl;
# endif
  lout << "Grids   :" << xstamp_grid_variants_as_string() << endl <<endl;

  // ============= random number generator state ==============
  // initialize random number generator
  if( ! rng_node.IsNull() )
  {
    exanb::rand::load_state( rng_node );
  }
  else 
  {
    exanb::rand::generate_seed();
    rng_node = exanb::rand::save_state();
//    config["random_generator_state"] = rng_node;
  }
  
  if( ! configuration.debug.rng.empty() )
  {
    std::ostringstream oss;
    oss << configuration.debug.rng << "." << rank;
    std::string file_name = oss.str();
    lout << "dump random state to file " <<file_name<< endl;    
    std::ofstream fout( file_name );
    dump_node_to_stream( fout, rng_node );
    fout << std::endl;
  }
  // ==========================================================
  

  // dump input files loaded
  if( configuration.debug.files )
  {
    lout << "===== loaded input files =====" << endl;
    for(string f : files_to_load) { lout << f << std::endl; }
    lout << "==============================" << endl << endl;
  }

  // dump input config
  if( configuration.debug.yaml )
  {
    lout << "======== configuration ========" << endl;
    dump_node_to_stream( lout, config_node );
    lout << std::endl << "==============================" << endl << endl;
    lout << "===== default definitions =====" << endl;
    dump_node_to_stream( lout, input_data );
    lout << std::endl << "==============================" << endl << endl;
    lout << "========= simulation ==========" << endl;
    dump_node_to_stream( lout, simulation_node );
    lout << std::endl << "==============================" << endl << endl;
  }
  if( configuration.debug.config )
  {
    lout << "======== configuration ========" << endl;
    configuration.m_doc.print_value( lout );
    lout << std::endl << "==============================" << endl << endl;
  }


  // ============= plugin loading ============
  
  if( configuration.plugin_db.empty() ) { configuration.plugin_db = configuration.plugin_dir + "/plugins_db.msp"; }
  if( configuration.debug.files ) { lout << "plugins search path is "<<configuration.plugin_dir << std::endl << std::endl; }
  
  // enable/disable verbosity for plugins load
  exanb::set_quiet_plugin_register( ! configuration.debug.plugins );

  // configure plugin DB generation if requested
  const PluginDBMap* plugin_db = nullptr;
  if( configuration.generate_plugins_db )
  {
    lout << "Writing plugins DB to " << configuration.plugin_db << std::endl;
    exanb::generate_plugin_db( configuration.plugin_db );
  }
  else
  {
    ldbg << "Reading plugins DB from " << configuration.plugin_db << std::endl;
    plugin_db = & exanb::read_plugin_db( configuration.plugin_db );
  }

  // load plugins and register builtin factories
  if( configuration.debug.plugins ) { lout << "============= plugins ===========" << endl << "+ <builtin>" << endl; }
  OperatorSlotBase::enable_registration();
  OperatorNodeFactory::instance()->enable_registration();
  exanb::set_default_plugin_search_dir( configuration.plugin_dir );
  if( ! configuration.plugins.empty() ) 
  {
    exanb::load_plugins( configuration.plugins , configuration.debug.plugins );
  }
  if( configuration.debug.plugins ) { lout << "=================================" << endl << endl; }

  plugins_loaded_breakpoint();
  
  if( configuration.generate_plugins_db )
  {
    return 0;
  }
  // =========================================


  // unit tests execution mode
  if( configuration.run_unit_tests )
  {
    auto [ n_passed , n_failed ] = UnitTest::run_unit_tests();
    lout << "Executed "<<(n_passed+n_failed)<<" unit tests : "<< n_passed << " passed, "<< n_failed <<" failed" << std::endl;
    return n_failed;
  }


  // ============ profiling configuration ==============
  OperatorNode::set_global_mem_profiling( configuration.profiling.resmem );
  bool configuration_needs_profiling = configuration.profiling.trace.enable || configuration.profiling.summary || configuration.profiling.exectime;
  OperatorNode::set_global_profiling( configuration_needs_profiling );
  onika::trace::TraceOutputFormat * otf = nullptr;
  if( configuration.profiling.trace.enable )
  {
    if( nb_procs > 1 )
    {
      std::ostringstream oss;
      oss << configuration.profiling.trace.file << "." << rank;
      configuration.profiling.trace.file = oss.str();
    }

    ViteColoringFunction colfunc = g_vite_operator_rnd_color;
    if( configuration.profiling.trace.color == "tag" ) colfunc = g_vite_tag_rnd_color;

    if( configuration.profiling.trace.format == "vite" )
    {
      onika::trace::ViteOutputFormat* votf = new onika::trace::ViteOutputFormat();
      otf = votf;
    }
    else if( configuration.profiling.trace.format == "dot" )
    {
      onika::trace::DotTraceFormat* dotf = new onika::trace::DotTraceFormat();
      otf = dotf;
    }
    else if( configuration.profiling.trace.format == "yaml" )
    {
      onika::trace::YAMLTraceFormat* yotf = new onika::trace::YAMLTraceFormat();
      otf = yotf;
    }
    else
    {
      lerr << "unsupported trace format '"<<configuration.profiling.trace.format <<"'"<<std::endl;
      std::abort();
    }

    vite_start_trace( configuration.profiling.trace, otf, g_vite_operator_label , colfunc );
    OperatorNode::set_profiler( { nullptr , vite_process_event } );
  }
  else if( configuration.profiling.exectime )
  {
# ifndef NDEBUG
    OperatorNode::set_profiler( { exanb::main::profiler_record_tag , exanb::main::log_profiler_stop_event } );
# else
    OperatorNode::set_profiler( { nullptr , exanb::main::log_profiler_stop_event } );
# endif
  }
# ifndef NDEBUG
  else
  {
    OperatorNode::set_global_profiling( true );
    OperatorNode::set_profiler( { exanb::main::profiler_record_tag , nullptr } );
  }
# endif
  // ===============================================
  

  // insert non configuration yaml to graph nodes to populate operators' default definitions
  OperatorNodeFactory::instance()->set_operator_defaults( input_data );

  // here is the good place to produce help generation if needed
  if( ! configuration.help.empty() )
  {
    lout << std::endl
         << "==============================" << std::endl
         << "============ help ============" << std::endl
         << "==============================" << std::endl << std::endl;
    if( configuration.help == "true" )
    {
      lout<<"Usage: "<<argv[0]<<" <input-file> [opt1,opt2...]"<<endl;
      lout<<"   Or: "<<argv[0]<<" --help default-config"<<endl;
      lout<<"   Or: "<<argv[0]<<" --help command-line"<<endl;
      lout<<"   Or: "<<argv[0]<<" --help plugins"<<endl<<endl;
      lout<<"   Or: "<<argv[0]<<" --help [operator-name]"<<endl<<endl;
#     ifndef NDEBUG
      lout<<"Debug with gdb -ex 'break simulation_start_breakpoint' -ex run --args"<<argv[0]<<endl<<endl;
#     endif
    }
    else if( configuration.help == "default-config" )
    {
      lout<<"Command line options:"<<endl
          <<"====================="<<endl<<endl;
      configuration.m_doc.print_default_config( lout );
      lout<<endl;
    }
    else if( configuration.help == "command-line" )
    {
      lout<<"default configuration:"<<endl
          <<"======================"<<endl<<endl;
      configuration.m_doc.print_command_line_options( lout );
      lout<<endl;
    }
    else if( configuration.help == "plugins" )
    {
      lout<<"Plugin data base:"<<endl
          <<"================="<<endl<<endl;
      std::set<std::string> available_plugins;
      std::map< std::string , std::set<std::string> > available_items;
      for(const auto& cp : *plugin_db )
      {
        for(const auto& op : cp.second )
        {
          available_items[cp.first].insert(op.first);
          available_plugins.insert( op.second );
        }
      }
      lout<<"Available plugins :"<<endl;
      for(const auto& s : available_plugins) { lout<<"\t"<<s<<std::endl; }
      
      for(const auto& cp : available_items)
      {
        lout<<endl<<"Available "<<cp.first<<"s :"<<endl;
        for(const auto& item : cp.second)
        {
          lout<<"\t"<<item<<std::endl;
        }
      }
    }
    else
    {
      std::shared_ptr<OperatorNode> op = OperatorNodeFactory::instance()->make_operator( configuration.help , YAML::Node(YAML::NodeType::Map) );
      std::ostringstream oss;
      op->print_documentation( oss );
      lout << oss.str() << std::endl;
    }
    return 0;
  }

  // prepare operator assembly strategy
  std::shared_ptr<OperatorNode> simulation_graph = OperatorNodeFactory::instance()->make_operator( "simulation" , simulation_node );  
  simulation_graph->apply_graph( [](OperatorNode* o){ o->post_graph_build(); } );

  //  print simulation graph
  if( configuration.debug.graph )
  {
    std::set<OperatorNode*> shrunk_nodes;
    for(const std::string& f : configuration.debug.graph_filter)
    {
      const std::regex re(f);
      simulation_graph->apply_graph(
        [&re,&shrunk_nodes](OperatorNode* op)
        {
          if( std::regex_match(op->pathname(),re) )
          {
            // std::cout <<"shrink graph node "<< op->pathname() << std::endl;
            shrunk_nodes.insert(op);
          }
        });
    }

    if( configuration.debug.graph_fmt == "console" )
    {
      lout<<endl<<"======= simulation graph ========"<<endl;
      simulation_graph->pretty_print(lout, configuration.debug.graph_lod );
      lout<<      "================================="<<endl<<endl;
    }
    else if( configuration.debug.graph_fmt == "dot" )
    {
      const std::string filename = "sim_graph.dot";
      bool show_unconnected_nodes = ( configuration.debug.graph_lod >= 2 );
      lout << "output simulation graph to "<<filename<<" ..." << std::endl;
      bool expose_batch_slots = true;
      bool remove_batch_ended_path = true;
      bool invisible_batch_slots = true;
      DotGraphOutput dotgen { expose_batch_slots, remove_batch_ended_path, invisible_batch_slots, configuration.debug.graph_rsc };
      dotgen.dot_sim_graph(simulation_graph.get(),filename, show_unconnected_nodes, shrunk_nodes);
    }
    else
    {
      lerr << "Unrecognized graph output format "<< configuration.debug.graph_fmt << std::endl;
      std::abort();
    }
  }

  // setup debug log filtering
  if( ! configuration.debug.filter.empty() )
  {
    //lout << "set debug filtering"<< std::endl;
    //for(const auto& f:configuration.debug.filter) lout << "\t" << f<< std::endl;
    auto hashes = operator_set_from_regex( simulation_graph, configuration.debug.filter, { { "ooo" , std::numeric_limits<uint64_t>::max() } } , "debug filter: add " );
    //lout << "hashes =>"<< std::endl;
    //for(auto h:hashes) lout << "\t" << h << std::endl;
    ldbg_raw.set_filters( hashes );
  }

  // print nodes' addresses
  if( configuration.debug.graph_addr )
  {
    std::set<OperatorNode*> op_addresses;
    simulation_graph->apply_graph( [&op_addresses](OperatorNode* op) { op_addresses.insert(op); });
    for(auto op:op_addresses) { std::cout<< (void*)op << " : " << op->name() << std::endl; }
  }

  // enable/disable additional debug messages for execution traces
  OperatorNode::set_debug_execution( configuration.debug.graph_exec );

  // activate verbose debugging of ompt tracing
  if( configuration.debug.ompt ) { onika::omp::OpenMPToolInterace::enable_internal_dbg_message(); }
  else { onika::omp::OpenMPToolInterace::disable_internal_dbg_message(); }

  // setup profiling filtering
  if( ! configuration.profiling.filter.empty() )
  {
    auto hashes = operator_set_from_regex( simulation_graph, configuration.profiling.filter, {} , "profiling enabled for " );
    simulation_graph->apply_graph(
      [&hashes](OperatorNode* o)
      {
        if( hashes.find(o->hash())==hashes.end() ) o->set_profiling(false);
      });
  }

  // print info about non profiled components
  if( configuration_needs_profiling )
  {
    std::set<std::string> noprof;
    simulation_graph->apply_graph( [&noprof](OperatorNode* op) { if(!op->profiling()) noprof.insert(op->pathname()); } );
    /*if( ! noprof.empty() )
    {
      lout << "profiling disabled components :"<<std::endl;
      for(auto op:noprof)
      {
        lout << "\t" << op << std::endl;
      }
    }*/
  }

  // setup GPU disable filtering
  if( ! configuration.onika.gpu_disable_filter.empty() )
  {
    auto hashes = operator_set_from_regex( simulation_graph, configuration.onika.gpu_disable_filter, {} , "GPU disabled for " );
    simulation_graph->apply_graph(
      [&hashes](OperatorNode* o)
      {
        if( hashes.find(o->hash())!=hashes.end() ) o->set_gpu_enabled(false);
      });
  }

  /**********************************/
  /***** Onika sub-sytem config *****/
  /**********************************/
  onika::parallel::ParallelExecutionContext::s_parallel_task_core_mult = configuration.onika.parallel_task_core_mult;
  onika::parallel::ParallelExecutionContext::s_parallel_task_core_add  = configuration.onika.parallel_task_core_add;
  onika::parallel::ParallelExecutionContext::s_gpu_sm_mult             = configuration.onika.gpu_sm_mult;
  onika::parallel::ParallelExecutionContext::s_gpu_sm_add              = configuration.onika.gpu_sm_add;
  onika::parallel::ParallelExecutionContext::s_gpu_block_size          = configuration.onika.gpu_block_size;

  /**********************************/
  /********* run simulation *********/
  /**********************************/
  exanb::OperatorNode::reset_profiling_reference_timestamp();
  if( configuration_needs_profiling )
  {
    onika::omp::OpenMPToolInterace::enable();
  }
  simulation_start_breakpoint();
  simulation_graph->run();
  if( configuration_needs_profiling )
  {
    onika::omp::OpenMPToolInterace::disable();
  }
  /**********************************/
  /**********************************/

  // produce vite trace output
  if( configuration.profiling.trace.enable )
  {
    vite_end_trace( configuration.profiling.trace );
    delete otf;
  }

  //  print simulation execution summary
  if( configuration.profiling.summary )
  {
    lout<<endl<<"Profiling .........................................  tot. time  ( GPU )   avginb  maxinb     count  percent"<<endl;        
    auto statsfunc = []( const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg )
    {
      exanb::mpi_parallel_stats(MPI_COMM_WORLD,x,np,r,minval,maxval,avg);
    };
    simulation_graph->pretty_print(lout,false,true,statsfunc);
    lout<<"=================================="<<endl<<endl;
  }

  // free all resources before exit
  simulation_graph->apply_graph( [](OperatorNode* op){ op->free_all_resources(); } );
  simulation_graph = nullptr;

  return 0;
}

