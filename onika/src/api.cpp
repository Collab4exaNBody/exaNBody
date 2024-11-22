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

#include <onika/app/api.h>
#include <fenv.h>

// dummy function to be used as a breakpoint marker just before simulation is ran
// usefull for adding breakpoints in loaded plugins
void simulation_start_breakpoint() {}
void plugins_loaded_breakpoint() {}

namespace onika
{
  namespace app
  {

    int ApplicationContext::get_error_code() const
    {
      return m_return_code;
    }
    
    onika::scg::OperatorNode* ApplicationContext::node(const std::string& nodepath) const
    {
      return node_from_path( m_simulation_graph , nodepath );
    }

    void ApplicationContext::set_multiple_run(bool yn)
    {
      m_simulation_graph->set_multiple_run( yn );
    }


    std::shared_ptr<ApplicationContext>
    init( int argc, char const * const argv[] )
    {
      std::shared_ptr<ApplicationContext> ctx = std::make_shared<ApplicationContext>();
  
#     ifndef NDEBUG
      std::cout << "to debug, use 'b simulation_start_breakpoint()' in gdb to stop program when all symbols are loaded"<<std::endl;
#     endif

      onika::app::initialize();
      auto [ main_input_files , cmdline ] = onika::app::parse_command_args( argc , argv );
      auto [ input_data , simulation_node , configuration ] = onika::app::load_yaml_input( main_input_files , cmdline );

      // ============= MPI Initialization =============
      const auto [ rank, nb_procs, mt_support, external_mpi_init ] = initialize_mpi( configuration , argc, argv /* , MPI_COMM_WORLD */ );

      // =========== OpenMP Initialization ============
      // third parameter may be be 'false' if configuration is driven by external application
      const auto [ cpucount , cpu_hw_threads , cpu_ids ] = onika::app::intialize_openmp( configuration , rank , true ); 

      // =========== configure logging system ===========
      onika::configure_logging(configuration.logging.debug, configuration.logging.parallel, configuration.logging.log_file, configuration.logging.err_file, configuration.logging.dbg_file, rank,nb_procs);
      
      // ============= GPU Initialization ===============
      int n_gpus = initialize_gpu( configuration );

      // ============= random number generator state ==============
      onika::parallel::generate_seed();

      // ============= system info ==============
      onika::app::print_host_system_info( onika::lout, configuration, nb_procs, cpucount, cpu_hw_threads, cpu_ids, n_gpus);

      // ============= plugins initialization ============  
      const auto [ plugin_db , may_run_simulation ] = onika::app::initialize_plugins( configuration );
      if( ! may_run_simulation ) { ctx->m_return_code=0; return ctx; }

      // ============= unit tests ============  
      const auto [ n_passed , n_failed ] = onika::app::run_embedded_tests( configuration );
      if( (n_passed+n_failed) > 0 ) { ctx->m_return_code=n_failed; return ctx; }

      // ============ profiling configuration ==============
      auto [ otf , configuration_needs_profiling ] = onika::app::initialize_profiling( configuration , rank, nb_procs );
      
      // insert non configuration yaml to graph nodes to populate operators' default definitions
      onika::scg::OperatorNodeFactory::instance()->set_operator_defaults( input_data );

      // print help if requested. if so, abort execution right after
      if( print_help(configuration,argv[0],plugin_db) ) { ctx->m_return_code=0; return ctx; }

      // prepare operator assembly strategy
      auto simulation_graph = onika::app::build_simulation_graph( configuration , simulation_node );
            
      ctx->m_configuration = std::make_shared<onika::app::ApplicationConfiguration>(configuration);
      ctx->m_input_files = main_input_files;
      ctx->m_cmdline_config = cmdline;
      ctx->m_input_data = input_data;
      ctx->m_simulation_node = simulation_node;
      ctx->m_cpucount = cpucount;
      ctx->m_cpu_hw_threads = cpu_hw_threads;
      ctx->m_cpu_ids = cpu_ids;
      ctx->m_mpi_rank = rank;
      ctx->m_mpi_nprocs = nb_procs;
      ctx->m_mpi_mt_support_level = mt_support;
      ctx->m_mpi_external_init = external_mpi_init;
      ctx->m_ngpus = n_gpus;
      ctx->m_test_npassed = n_passed;
      ctx->m_test_nfailed = n_failed;
      ctx->m_prof_trace = otf;
      ctx->m_need_profiling = configuration_needs_profiling;
      ctx->m_plugin_db = plugin_db;
      ctx->m_plugin_db_generate_mode = ! may_run_simulation;
      ctx->m_simulation_graph = simulation_graph;
      
      return ctx;
    }

    void
    run(std::shared_ptr<ApplicationContext> ctx)
    {
      onika::app::run_simulation_graph( ctx->m_simulation_graph , ctx->m_need_profiling );
    }

    void
    end(std::shared_ptr<ApplicationContext> ctx)
    {
      onika::app::finalize( * (ctx->m_configuration.get()) , ctx->m_simulation_graph , ctx->m_prof_trace );
    }
    
    void
    initialize()
    {
      const char * confpath = std::getenv("ONIKA_CONFIG_PATH");
      if( confpath != nullptr )
      {
        std::cout<<"set config path from ONIKA_CONFIG_PATH env to '"<<confpath<<"'"<<std::endl;
        set_install_config_dir( confpath );
      }

      const char * datapath = std::getenv("ONIKA_DATA_PATH");
      if( datapath != nullptr ) set_data_file_dirs( datapath );
    }


    void
    finalize( const onika::app::ApplicationConfiguration & configuration
                        , std::shared_ptr<onika::scg::OperatorNode> simulation_graph
                        , onika::trace::TraceOutputFormat * otf )
    {
      using namespace onika::scg;
    
      // produce vite trace output
      if( configuration.profiling.trace.enable )
      {
        vite_end_trace( configuration.profiling.trace );
        delete otf;
      }

      //  print simulation execution summary
      if( configuration.profiling.summary )
      {
        lout<<std::endl<<"Profiling .........................................  tot. time  ( GPU )   avginb  maxinb     count  percent"<<std::endl;        
        auto statsfunc = []( const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg )
        {
          onika::mpi::mpi_parallel_stats(MPI_COMM_WORLD,x,np,r,minval,maxval,avg);
        };
        simulation_graph->pretty_print(lout,false,true,statsfunc);
        lout<<"=================================="<<std::endl<<std::endl;
      }

      // free all resources before exit
      simulation_graph->apply_graph( [](OperatorNode* op){ op->free_all_resources(); } );
      simulation_graph = nullptr;
    }

    std::pair< std::vector<std::string> , YAML::Node >
    parse_command_args( int argc, char const * const argv[] )
    {
      if( argc < 2 )
      {
        lerr<<"Usage: "<<argv[0]<<" <input-file> [opt1,opt2...]"<<std::endl;
        lerr<<"   Or: "<<argv[0]<<" --help [operator-name]"<<std::endl;
        std::exit(1);
      }

      std::vector<std::string> main_input_files;
      YAML::Node cmdline(YAML::NodeType::Map);

      int start_opt_arg = 1;
      main_input_files.clear();
      while(start_opt_arg < argc && std::string(argv[start_opt_arg]).find("--")!=0 )
      {
        main_input_files.push_back( argv[start_opt_arg] );
        ++ start_opt_arg;
      }

      // additional arguments are interpreted as YAML strings that are parsed, and merged on top of files read previously
      onika::yaml::command_line_options_to_yaml_config(argc,argv,start_opt_arg,cmdline);
      return { main_input_files , cmdline };
    }



    std::tuple<YAML::Node,YAML::Node,onika::app::ApplicationConfiguration>
    load_yaml_input( const std::vector<std::string>& main_input_files, YAML::Node cmdline )
    {
      using namespace onika;
      using namespace onika::scg;

      // load user file and all subsequent include includes.
      // when no includes is specified, USTAMP_DEFAULT_CONFIG_FILE is loaded as if it has been included.
      // to prevent any file from being included, write "includes: []" in your input file
      std::vector<std::string> files_to_load = onika::yaml::resolve_config_file_includes( main_input_files );
      assert( ! files_to_load.empty() );

      // merge YAML nodes from inner most included files up to user provided file
      YAML::Node input_data(YAML::NodeType::Map);
      for(auto f:files_to_load)
      {
        std::string pf = onika::config_file_path( f );
        //onika::ldbg
        std::cout << "load config file "<< pf << std::endl; ldbg << std::flush;
        input_data = onika::yaml::merge_nodes( YAML::Clone(input_data) , onika::yaml::yaml_load_file_abort_on_except(pf) );
      }

      input_data = onika::yaml::merge_nodes( YAML::Clone(input_data) , cmdline );

      // ====== extract YAML information blocks =========
      YAML::Node config_node;
      if( input_data["configuration"] )
      {
        config_node = input_data["configuration"];
        input_data = onika::yaml::remove_map_key( input_data, "configuration" );
      }

      // convert YAML configuration node to data structure
      onika::app::ApplicationConfiguration configuration { config_node };

      // allow special block configuration block "set" to overload base input data
      if( configuration.set.IsMap() && configuration.set.size()>0 )
      {
        input_data = onika::yaml::merge_nodes( input_data , configuration.set );
        configuration.set = YAML::Node();
        config_node = onika::yaml::remove_map_key( config_node, "set" );
      }
      
      // simulation definition
      YAML::Node simulation_node;
      if( input_data["simulation"] )
      {
        simulation_node = input_data["simulation"];
        input_data = onika::yaml::remove_map_key( input_data, "simulation" );
      }

      // ======== process debugging options =============
      OperatorNodeFactory::set_debug_verbose_level( configuration.debug.verbose );
      if( configuration.debug.fpe ) { feenableexcept( FE_ALL_EXCEPT & ~FE_INEXACT ); }
      // ===============================================

      // dump input files loaded
      if( configuration.debug.files )
      {
        lout << "===== loaded input files =====" << std::endl;
        for(std::string f : files_to_load) { lout << f << std::endl; }
        lout << "==============================" << std::endl << std::endl;
      }

      // dump input config
      if( configuration.debug.yaml )
      {
        lout << "======== configuration ========" << std::endl;
        onika::yaml::dump_node_to_stream( lout, config_node );
        lout << std::endl << "==============================" << std::endl << std::endl;
        lout << "===== default definitions =====" << std::endl;
        onika::yaml::dump_node_to_stream( lout, input_data );
        lout << std::endl << "==============================" << std::endl << std::endl;
        lout << "========= simulation ==========" << std::endl;
        onika::yaml::dump_node_to_stream( lout, simulation_node );
        lout << std::endl << "==============================" << std::endl << std::endl;
      }
      if( configuration.debug.config )
      {
        lout << "======== configuration ========" << std::endl;
        configuration.m_doc.print_value( lout );
        lout << std::endl << "==============================" << std::endl << std::endl;
      }
      
      return { input_data , simulation_node , configuration };
    }
    
    
    
    onika::scg::OperatorNode*
    node_from_path( std::shared_ptr<onika::scg::OperatorNode> simulation_graph , const std::string& nodepath )
    {
      onika::scg::OperatorNode* node = nullptr;
      simulation_graph->apply_graph( [&nodepath,&node](onika::scg::OperatorNode* o){ if( o->pathname() == nodepath ) node = o; } );
      return node;
    }


    std::tuple< int , int , std::vector<int> >
    intialize_openmp( onika::app::ApplicationConfiguration & configuration , int mpi_rank , bool allow_openmp_conf )
    {
      // ============= OpenMP Initialization =============
      if( configuration.omp_num_threads > 0 && allow_openmp_conf )
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
      
      if( configuration.omp_max_nesting > 1 && allow_openmp_conf )
      {
        if(configuration.omp_nested) { omp_set_nested(1); }
        omp_set_max_active_levels( configuration.omp_max_nesting );
      }
      configuration.omp_max_nesting = omp_get_max_active_levels();
      
      int num_threads = 0;
#     pragma omp parallel
      {
#       pragma omp single
        {
          num_threads = omp_get_num_threads();
        }
      }
      configuration.omp_num_threads = num_threads;
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
        if( allow_openmp_conf ) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff);
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff);
        /***************/

        int ncpus = CPU_COUNT(&cpuaff);

        /* restore initial mask */    
        if( allow_openmp_conf ) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuaff_backup);
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
          fatal_error() << "Internal error when counting CPU cores" << std::endl;
        }
        if( cpucount<num_threads && configuration.pinethreads && allow_openmp_conf )
        {
          lerr << "Thread pining disabled because there are less cpu cores ("<<cpucount<<") than OpenMP threads ("<<num_threads<<")" << std::endl;
          configuration.pinethreads = false;
        }
      }
      // get the real core count (not the cpumask)
      int cpu_hw_threads = std::thread::hardware_concurrency();

      // =========== optional thread pinning ===========
      if( configuration.pinethreads )
      {
#       pragma omp parallel
        {
          cpu_set_t thread_cpu_aff;
          CPU_ZERO(&thread_cpu_aff);
          int tid = omp_get_thread_num();
          CPU_SET(cpu_ids[ ( tid + mpi_rank * configuration.threadrotate ) % cpu_ids.size() ], &thread_cpu_aff);
          pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &thread_cpu_aff);
        }
      }
      // ===============================================
      
      onika::parallel::ParallelExecutionContext::s_parallel_task_core_mult = configuration.onika.parallel_task_core_mult;
      onika::parallel::ParallelExecutionContext::s_parallel_task_core_add  = configuration.onika.parallel_task_core_add;

      return { cpucount , cpu_hw_threads , cpu_ids };
    }
    
    
    
    std::string
    cpu_id_list_to_string(const std::vector<int>& cpu_ids)
    {
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
      return core_config;
    }
    
    std::tuple<int,int,int,int>
    initialize_mpi( const onika::app::ApplicationConfiguration & configuration , int argc, char const * const in_argv[], MPI_Comm app_world_comm )
    {
      std::vector<char*> argv_vec(argc,nullptr);
      for(int a=0;a<argc;a++) argv_vec[a] = strdup(in_argv[a]);
      char ** argv = argv_vec.data();

      // ============= MPI Initialization =============
      int rank=0, nb_procs=0;
      int mt_support = 0;
      int external_mpi_init = 0;
      MPI_Initialized( &external_mpi_init );
      
      if( external_mpi_init )
      {
        MPI_Query_thread( &mt_support );
      }
      else
      {
        if( configuration.mpimt )
        {
          MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mt_support);
        }
        else
        {          
          MPI_Init(&argc, &argv);
        }
      }
      for(int a=0;a<argc;a++) free( argv_vec[a] );
      argv_vec.clear(); argv=nullptr;
      
      MPI_Comm_rank(app_world_comm, &rank);
      MPI_Comm_size(app_world_comm, &nb_procs);

      // scoped variable that properly finalizes MPI upon main function exit
      struct MpiScopedFinalizer { bool finalize_on_exit=true; ~MpiScopedFinalizer() { if(finalize_on_exit) MPI_Finalize(); } } mpi_finalize_on_scope_exit = { !external_mpi_init };
      if( configuration.mpimt && mt_support != MPI_THREAD_MULTIPLE && rank==0 )
      {
        lerr<<"Warning: no MPI_THREAD_MULTIPLE support. Concurrent calls to mpi API will fail."<<std::endl;
      }
      // ===============================================
      return { rank, nb_procs, mt_support, external_mpi_init };
    }
    
    
    
    int
    initialize_gpu( const onika::app::ApplicationConfiguration & configuration )
    {
      // get number of available GPUs, if any   
      int n_gpus = 0;
#     ifdef ONIKA_CUDA_VERSION
      onika::cuda::CudaContext::set_global_gpu_enable( ! configuration.nogpu );
      if( onika::cuda::CudaContext::global_gpu_enable() )
      {
        auto cu_dev_count_rc = ONIKA_CU_GET_DEVICE_COUNT(&n_gpus);
        if( n_gpus > 0 )
        {
          ONIKA_CU_CHECK_ERRORS( cu_dev_count_rc );
        }
      }
#     else
      n_gpus = 0;
#     endif
      if( n_gpus == 0 ) { onika::cuda::CudaContext::set_global_gpu_enable( false ); }
      onika::memory::GenericHostAllocator::set_cuda_enabled( n_gpus > 0 );

      onika::parallel::ParallelExecutionContext::s_gpu_sm_mult    = configuration.onika.gpu_sm_mult;
      onika::parallel::ParallelExecutionContext::s_gpu_sm_add     = configuration.onika.gpu_sm_add;
      onika::parallel::ParallelExecutionContext::s_gpu_block_size = configuration.onika.gpu_block_size;

      return n_gpus;
    }
      
      
      
    std::pair< const onika::PluginDBMap* , bool >
    initialize_plugins( onika::app::ApplicationConfiguration & configuration )
    {
      using namespace onika::scg;

      if( configuration.plugin_db.empty() ) { configuration.plugin_db = configuration.plugin_dir + "/plugins_db.msp"; }
      if( configuration.debug.files ) { lout << "plugins search path is "<<configuration.plugin_dir << std::endl << std::endl; }
      
      // enable/disable verbosity for plugins load
      onika::set_quiet_plugin_register( ! configuration.debug.plugins );

      // configure plugin DB generation if requested
      const onika::PluginDBMap* plugin_db = nullptr;
      if( configuration.generate_plugins_db )
      {
        lout << "Writing plugins DB to " << configuration.plugin_db << std::endl;
        onika::generate_plugin_db( configuration.plugin_db );
      }
      else
      {
        ldbg << "Reading plugins DB from " << configuration.plugin_db << std::endl;
        plugin_db = & onika::read_plugin_db( configuration.plugin_db );
      }

      // load plugins and register builtin factories
      if( configuration.debug.plugins ) { lout << "============= plugins ===========" << std::endl << "+ <builtin>" << std::endl; }
      OperatorSlotBase::enable_registration();
      OperatorNodeFactory::instance()->enable_registration();
      onika::set_default_plugin_search_dir( configuration.plugin_dir );
      if( ! configuration.plugins.empty() ) 
      {
        onika::load_plugins( configuration.plugins , configuration.debug.plugins );
      }
      if( configuration.debug.plugins ) { lout << "=================================" << std::endl << std::endl; }

      plugins_loaded_breakpoint();
      
      return { plugin_db , ! configuration.generate_plugins_db };
    }
    
    
    
    std::pair<int,int> 
    run_embedded_tests( const onika::app::ApplicationConfiguration & configuration )
    {
      if( configuration.run_unit_tests )
      {
        auto [ n_passed , n_failed ] = onika::UnitTest::run_unit_tests();
        lout << "Executed "<<(n_passed+n_failed)<<" unit tests : "<< n_passed << " passed, "<< n_failed <<" failed" << std::endl;
        return { n_passed , n_failed };
      }
      else return { 0 , 0 };
    }



    std::pair< onika::trace::TraceOutputFormat * , bool >
    initialize_profiling( onika::app::ApplicationConfiguration & configuration, int mpi_rank, int nb_procs )
    {
      using namespace onika::scg;

      OperatorNode::set_global_mem_profiling( configuration.profiling.resmem );
      bool configuration_needs_profiling = configuration.profiling.trace.enable || configuration.profiling.summary || configuration.profiling.exectime;
      OperatorNode::set_global_profiling( configuration_needs_profiling );
      onika::trace::TraceOutputFormat * otf = nullptr;
      if( configuration.profiling.trace.enable )
      {
        if( nb_procs > 1 )
        {
          std::ostringstream oss;
          oss << configuration.profiling.trace.file << "." << mpi_rank;
          configuration.profiling.trace.file = oss.str();
        }

        onika::app::ViteColoringFunction colfunc = onika::app::g_vite_operator_rnd_color;
        if( configuration.profiling.trace.color == "tag" ) colfunc = onika::app::g_vite_tag_rnd_color;

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
          fatal_error() << "unsupported trace format '"<<configuration.profiling.trace.format <<"'"<<std::endl;
        }

        onika::app::vite_start_trace( configuration.profiling.trace, otf, onika::app::g_vite_operator_label , colfunc );
        OperatorNode::set_profiler( { nullptr , onika::app::vite_process_event } );
      }
      else if( configuration.profiling.exectime )
      {
#       ifndef NDEBUG
        OperatorNode::set_profiler( { onika::app::profiler_record_tag , onika::app::log_profiler_stop_event } );
#       else
        OperatorNode::set_profiler( { nullptr , onika::app::log_profiler_stop_event } );
#       endif
      }
#     ifndef NDEBUG
      else
      {
        OperatorNode::set_global_profiling( true );
        OperatorNode::set_profiler( { onika::app::profiler_record_tag , nullptr } );
      }
#     endif

      return { otf , configuration_needs_profiling };
    }



    bool 
    print_help( const onika::app::ApplicationConfiguration & configuration , const std::string& appname, const onika::PluginDBMap* plugin_db )
    {
      using namespace onika::scg;

      // here is the good place to produce help generation if needed
      if( ! configuration.help.empty() )
      {
        lout << std::endl
             << "==============================" << std::endl
             << "============ help ============" << std::endl
             << "==============================" << std::endl << std::endl;
        if( configuration.help == "true" )
        {
          lout<<"Usage: "<<appname<<" <input-file> [opt1,opt2...]"<<std::endl;
          lout<<"   Or: "<<appname<<" --help default-config"<<std::endl;
          lout<<"   Or: "<<appname<<" --help command-line"<<std::endl;
          lout<<"   Or: "<<appname<<" --help plugins"<<std::endl<<std::endl;
          lout<<"   Or: "<<appname<<" --help show-plugins"<<std::endl<<std::endl;
          lout<<"   Or: "<<appname<<" --help [operator-name]"<<std::endl<<std::endl;
#         ifndef NDEBUG
          lout<<"Debug with gdb -ex 'break simulation_start_breakpoint' -ex run --args"<<appname<<std::endl<<std::endl;
#         endif
        }
        else if( configuration.help == "default-config" )
        {
          lout<<"Command line options:"<<std::endl
              <<"====================="<<std::endl<<std::endl;
          configuration.m_doc.print_default_config( lout );
          lout<<std::endl;
        }
        else if( configuration.help == "command-line" )
        {
          lout<<"default configuration:"<<std::endl
              <<"======================"<<std::endl<<std::endl;
          configuration.m_doc.print_command_line_options( lout );
          lout<<std::endl;
        }
        else if( configuration.help == "show-plugins" )
        {
          lout<<"Operator data base:"<<std::endl
              <<"================="<<std::endl<<std::endl;
          std::ostringstream oss;
          for(const auto& cp : *plugin_db )
          {
            for(const auto& op : cp.second )
            {
              if( op.first == "batch") continue;
              if( op.first == "matrix_4d") continue;
              if( op.first == "particle_region_csg") continue;
              std::shared_ptr<OperatorNode> ope = OperatorNodeFactory::instance()->make_operator( op.first , YAML::Node(YAML::NodeType::Map ) );
              ope->print_documentation( oss );
            }
          }
          lout << oss.str() << std::endl;
        }
        else if( configuration.help == "plugins" )
        {
          lout<<"Plugin data base:"<<std::endl
            <<"================="<<std::endl<<std::endl;
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
          lout<<"Available plugins :"<<std::endl;
          for(const auto& s : available_plugins) { lout<<"\t"<<s<<std::endl; }

          for(const auto& cp : available_items)
          {
            lout<<std::endl<<"Available "<<cp.first<<"s :"<<std::endl;
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
        return true;
      }
      else return false;
    }



    std::shared_ptr<onika::scg::OperatorNode>
    build_simulation_graph( const onika::app::ApplicationConfiguration & configuration , YAML::Node simulation_node )
    {
      using namespace onika::scg;

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
          lout<<std::endl<<"======= simulation graph ========"<<std::endl;
          simulation_graph->pretty_print(lout, configuration.debug.graph_lod );
          lout<<      "================================="<<std::endl<<std::endl;
        }
        else if( configuration.debug.graph_fmt == "dot" )
        {
          const std::string filename = "sim_graph.dot";
          bool show_unconnected_nodes = ( configuration.debug.graph_lod >= 2 );
          lout << "output simulation graph to "<<filename<<" ..." << std::endl;
          bool expose_batch_slots = true;
          bool remove_batch_ended_path = true;
          bool invisible_batch_slots = true;
          onika::app::DotGraphOutput dotgen { expose_batch_slots, remove_batch_ended_path, invisible_batch_slots, configuration.debug.graph_rsc };
          dotgen.dot_sim_graph(simulation_graph.get(),filename, show_unconnected_nodes, shrunk_nodes);
        }
        else
        {
          fatal_error() << "Unrecognized graph output format "<< configuration.debug.graph_fmt << std::endl;
        }
      }

      // setup debug log filtering
      if( ! configuration.debug.filter.empty() )
      {
        //lout << "set debug filtering"<< std::endl;
        //for(const auto& f:configuration.debug.filter) lout << "\t" << f<< std::endl;
        auto hashes = onika::scg::operator_set_from_regex( simulation_graph, configuration.debug.filter, { { "ooo" , std::numeric_limits<uint64_t>::max() } } , "debug filter: add " );
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
        auto hashes = onika::scg::operator_set_from_regex( simulation_graph, configuration.profiling.filter, {} , "profiling enabled for " );
        simulation_graph->apply_graph(
            [&hashes](OperatorNode* o)
            {
            if( hashes.find(o->hash())==hashes.end() ) o->set_profiling(false);
            });
      }

      // setup GPU disable filtering
      if( ! configuration.onika.gpu_disable_filter.empty() )
      {
        auto hashes = onika::scg::operator_set_from_regex( simulation_graph, configuration.onika.gpu_disable_filter, {} , "GPU disabled for " );
        simulation_graph->apply_graph(
            [&hashes](OperatorNode* o)
            {
              if( hashes.find(o->hash())!=hashes.end() ) o->set_gpu_enabled(false);
            });
      }
      if( ! configuration.onika.gpu_enable_filter.empty() )
      {
        auto hashes = onika::scg::operator_set_from_regex( simulation_graph, configuration.onika.gpu_enable_filter, {} , "GPU enabled for " );
        simulation_graph->apply_graph(
            [&hashes](OperatorNode* o)
            {
              if( hashes.find(o->hash())!=hashes.end() ) o->set_gpu_enabled(true);
            });
      }

      // setup OpenMP threads limitations for filtered operators
      if( ! configuration.omp_max_threads_filter.empty() )
      {
        for(const auto& p : configuration.omp_max_threads_filter )
        {
          const int nthreads = p.second;
          auto hashes = onika::scg::operator_set_from_regex( simulation_graph, { p.first } );
          simulation_graph->apply_graph(
              [&hashes,nthreads](OperatorNode* o)
              {
                if( hashes.find(o->hash())!=hashes.end() ) { ldbg<<"Limit maximum number of threads to "<<nthreads<<" for operator "<<o->pathname()<<std::endl; o->set_omp_max_threads(nthreads); }
              });
        }    
      }
      
      return simulation_graph;
    }



    void
    run_simulation_graph( std::shared_ptr<onika::scg::OperatorNode> simulation_graph , bool configuration_needs_profiling )
    {
      using namespace onika::scg;

      OperatorNode::reset_profiling_reference_timestamp();
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
    }
   
  }
}

