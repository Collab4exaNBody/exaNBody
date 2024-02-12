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

#include <exanb/core/operator_slot_base.h>
#include <exanb/core/operator_slot_direction.h>
#include <exanb/core/log.h>
#include <exanb/core/yaml_utils.h>
#include <exanb/core/span.h>

#include <yaml-cpp/yaml.h>

#include <map>
#include <set>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <type_traits>
#include <cstdlib>
#include <cassert>
#include <atomic>
#include <deque>

#include <onika/omp/ompt_task_timing.h>
#include <onika/omp/ompt_task_info.h>
#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>

namespace exanb
{
  struct OperatorNode;
  using OperatorNodeFlavor = std::map< std::string , std::string >;
  using OperatorNodeCreateFunction = std::function<std::shared_ptr<OperatorNode>(const YAML::Node&, const OperatorNodeFlavor&)>;
  
  using ParallelValueStatsFunc = std::function< void(const std::vector<double>&,int&,int&,std::vector<double>&,std::vector<double>&,std::vector<double>&) >;
  void default_parallel_stats_func(const std::vector<double>& x, int& np, int& r, std::vector<double>& minval, std::vector<double>& maxval, std::vector<double>& avg);

  using ProfilingFunction = std::function<void(const onika::omp::OpenMPToolTaskTiming&)>;
  struct ProfilingFunctionSet
  {
    ProfilingFunction task_start;
    ProfilingFunction task_stop;
  };

  // base class for compute operators
  // it works but it looks ugly. it needs cleanup
  struct OperatorNode
  {

    // encapsulates documentaiton strings used for slots
    struct DocString
    {
      std::string m_doc;
    };

    // helper that replaces ldbg with a filtered version of the log stream
    struct OperatorDebugStreamHelper
    {
      OperatorNode* m_op = nullptr;
      inline OperatorDebugStreamHelper(OperatorNode* op) : m_op(op) {}
      template<typename T>
      inline LogStreamWrapper& operator << (const T& x)
      {
        assert(m_op!=nullptr);
        return ldbg_raw.filter(m_op->hash()) << x;
      }
      
      inline LogStreamWrapper& operator << ( std::ostream& (*manip)(std::ostream&) )
      {
        assert(m_op!=nullptr);
        return ldbg_raw.filter(m_op->hash()) << manip ;
      }
    };

    // symbolic name that may be used to mark slot as required (they don not allow not having a value attached to it)
    struct REQUIRED_t {};
    static constexpr REQUIRED_t REQUIRED = REQUIRED_t();

    // symbolic name that may be used to mark slot as optional (allowed not to be populated with a value)
    struct OPTIONAL_t {};
    static constexpr OPTIONAL_t OPTIONAL = OPTIONAL_t();

    // symbolic name for private slots, aka INPUT_OUTPUT slot with constructible type not allowed to be connected to anything
    struct PRIVATE_t {};
    static constexpr PRIVATE_t PRIVATE = PRIVATE_t();

    // configure how to print (or not to print) profiling information
    struct ProfilePrintParameters
    {
      double m_total_time = 0.0;
      double m_inner_loop_time = 0.0;
      bool m_print_profiling = false;
    };

    // System timestamp type, used for profiling purposes
    using TimeStampT = decltype( std::chrono::high_resolution_clock::now() );

    // unique constrtuctor is the default one
    OperatorNode() = default;
    virtual ~OperatorNode();
    
    // to be called when operator will no longer be executed, before it is destructed
    virtual void finalize();

    // ==== main interface, executes the operator =======

    // called from outside, calls execute internally
    virtual void run_prolog();
    virtual void run(); 
    virtual void run_epilog();
    virtual void execute() =0;
    
    virtual inline bool nested_parallel_mode() const { return false; }
    virtual bool is_looping() const;
    
    virtual inline std::string documentation() const { return std::string(); }
    
    // print operator and its sub graph with various level of details
    inline LogStreamWrapper& pretty_print(LogStreamWrapper& out, int details, bool prof = false, ParallelValueStatsFunc pstat = default_parallel_stats_func )
    {
      ProfilePrintParameters p{0.,0.,prof};
      return pretty_print(out,details,0,p,pstat);
    }
    virtual LogStreamWrapper& pretty_print(LogStreamWrapper& out, int details, int indent, ProfilePrintParameters& ppp, ParallelValueStatsFunc pstat );
    
    // populate user given inputs from YAML config file
    virtual void yaml_initialize(const YAML::Node&);
    
    // tells if operator is terminal (true) or is a batch operator which holds other operators (false)
    virtual bool is_terminal() const;
    
    // a terminal operator that cannot be removed even though it has no connected output
    // more generally, an operator that have a side effect (writes a file, etc.) might return true also
    virtual bool is_sink() const;
    
    // apply a function to all operators in graph    
    virtual void apply_graph( std::function<void(OperatorNode*)> , bool prefix=false);

    // those 2 are virtual cause we need a specific implementation for batches whose slots are added afterward
    virtual OperatorNodeFlavor in_flavor() const;
    virtual OperatorNodeFlavor out_flavor() const;

    // what to do when the operator graph is complete
    virtual void post_graph_build();

    // eventually optimizes things. after a call to this function, no slots can be added or removed
    // slots names and operator name cannot be changed anymore. no sub operators (for batches) can change neither
    // thus, outter slot connections can be done later on.
    virtual void compile();

    // virtual std::shared_ptr<OperatorNode> clone() const;
    // ==================

    // indicates that an operator or batch is ready to be executed
    inline bool compiled() const { return m_compiled; }

    // ensures slot attached resources are allocated and initialized
    void initialize_slots_resource();

    // activate per operator profiling
    void set_profiling(bool prof);

    // additional debug messages for execution start/stop
    static void set_debug_execution(bool yn);
    static bool debug_execution();

    // operator naming
    const std::string& name() const;
    std::string pathname() const;
    void set_name(const std::string& n);

    // set the containing operator
    void set_parent( OperatorNode* parent );
    inline OperatorNode* parent() const { return m_parent; }

    // access the map key -> slots
    inline const std::unordered_set<OperatorSlotBase*>& slots() const { return m_slots; }
    std::set< std::pair<std::string,OperatorSlotBase*> > named_slots() const;

    // read only access to slots
    inline auto in_slots() const { return make_const_span(m_in_slot_storage,in_slot_count()); }
    inline auto out_slots() const { return make_const_span(m_out_slot_storage,out_slot_count()); }

    inline int in_slot_count() const { return m_in_slot_count; }
    int in_slot_idx(const std::string& k) const;
    int in_slot_idx(const OperatorSlotBase *s) const;
    bool in_slot_rename(const std::string& before, const std::string& after);
    inline OperatorSlotBase* in_slot(int i) const { if(i<0 || i>=in_slot_count()) return nullptr; else return m_in_slot_storage[i].second; }
    inline OperatorSlotBase* in_slot(const std::string& k) const { return in_slot(in_slot_idx(k)); }
    
    int out_slot_count() const { return m_out_slot_count; }
    int out_slot_idx(const std::string& k) const;
    int out_slot_idx(const OperatorSlotBase *s) const;
    bool out_slot_rename(const std::string& before, const std::string& after);
    OperatorSlotBase* out_slot(int i) const { if(i<0 || i>=out_slot_count()) return nullptr; else return m_out_slot_storage[i].second; }
    inline OperatorSlotBase* out_slot(const std::string& k) const { return out_slot(out_slot_idx(k)); }

    // what operators can be reached following the output data flow
    void outflow_reachable_slots(std::set<const OperatorSlotBase*>& ops) const;

    // extract operator's flavor, i.e. preferred type for it's slots
    OperatorNodeFlavor flavor() const;

    // global (all operators) and local (this operator only) profiling activation
    bool profiling() const;
    static void set_global_profiling(bool prof);
    static bool global_profiling();

    static void set_global_mem_profiling(bool prof);
    static bool global_mem_profiling();

    // register a slot created on its own ( not through add_slot<T> method )
    void register_slot( const std::string& name, OperatorSlotBase* s );
    void register_in_slot( const std::string& name, OperatorSlotBase* s );
    void register_out_slot( const std::string& name, OperatorSlotBase* s );
    void register_managed_slot( std::shared_ptr<OperatorSlotBase> s );

    // how deep if the operator in operator tree
    inline unsigned int depth() const { return m_depth; }

    // get operator hash    
    inline size_t hash() const { return m_hash; }
    static inline size_t max_hash() { return s_global_instance_index; }

    // register a profiling function pair
    static void set_profiler( ProfilingFunctionSet profiler );
    void profile_task_start( const onika::omp::OpenMPToolTaskTiming& evt_info );
    void profile_task_stop( const onika::omp::OpenMPToolTaskTiming& evt_info );

    // access GPUExecution context for this operator
    void set_gpu_enabled(bool en);
    // Warning! : this methods has a wrong name, for backward compatibility purposes

    // each call allocates a new context to be used to build up a new parallel operation
    void set_task_group_mode( bool m );
    bool task_group_mode() const;
    onika::parallel::ParallelExecutionContext* parallel_execution_context();
    std::shared_ptr<onika::parallel::ParallelExecutionStream> parallel_execution_stream_nolock(unsigned int id=0);
    std::shared_ptr<onika::parallel::ParallelExecutionStream> parallel_execution_stream_lock(unsigned int id=0);
    onika::parallel::ParallelExecutionStreamQueue parallel_execution_stream(unsigned int id=0);
    void wait_all_parallel_execution_streams();
    
    // free resources associated to slots
    void free_all_resources();

    // pretty print full documentation of operator and its 
    std::ostream& print_documentation( std::ostream& out ) const;

    // set wall clock start point for profiling measures
    static void reset_profiling_reference_timestamp();

    ssize_t resident_memory_inc() const;

  protected:
    inline const std::set< std::shared_ptr<OperatorSlotBase> >& managed_slots() const { return m_managed_slots; }

    static void set_global_cuda_ctx( std::shared_ptr<onika::cuda::CudaContext> ctx );
    static onika::cuda::CudaContext* global_cuda_ctx();

    // profiling
    std::atomic<uint64_t> m_task_exec_time_accum {0};
    std::vector<double> m_exec_times;
    std::vector<double> m_gpu_times;
    std::vector<double> m_async_cpu_times;
    TimeStampT m_run_start_time;
    double m_total_gpu_time = 0.0;
    double m_total_async_cpu_time = 0.0;
    ssize_t m_resident_mem = 0;
    ssize_t m_resident_mem_inc = 0;

    // debug stream wrapper to enable log filtering
    OperatorDebugStreamHelper ldbg { this };    

  private:

    static void finalize_parallel_execution(onika::parallel::ParallelExecutionContext* pec, void * v_self);
    static void task_start_callback( const onika::omp::OpenMPToolTaskTiming& evt );
    static void task_stop_callback( const onika::omp::OpenMPToolTaskTiming& evt );
  
    static constexpr size_t MAX_SLOT_COUNT = 128;
  
    // operator unique id
    size_t m_hash = 0;
    static size_t s_global_instance_index;

    // data flow slots
    size_t m_in_slot_count = 0;
    size_t m_out_slot_count = 0;
    std::pair<std::string,OperatorSlotBase*> m_in_slot_storage[MAX_SLOT_COUNT];
    std::pair<std::string,OperatorSlotBase*> m_out_slot_storage[MAX_SLOT_COUNT];

    // operator hierarchy
    size_t m_depth = 0;
    std::string m_name;
    std::unique_ptr<char[]> m_tag = nullptr; // a unique string pointer equivalent to pathname()
    OperatorNode* m_parent = nullptr;

    std::string m_backtrace; // stored to facilitate debugging

    // auxiliary slot registration
    std::unordered_set<OperatorSlotBase*> m_slots;
    std::set< std::shared_ptr<OperatorSlotBase> > m_managed_slots; // for proper deallocation

    // Parallel execution context and streams:
    // contains necessary resources to globalize parallel executions accross diferent operators and allow asynchornous parallel executions.
    std::mutex m_parallel_execution_access;
    OperatorNode* m_task_group_ancestor = nullptr; // highest ancestor creator of the encapsulating task group
    std::vector< std::shared_ptr<onika::parallel::ParallelExecutionStream> > m_parallel_execution_streams;
    std::vector< onika::parallel::ParallelExecutionContext* > m_free_parallel_execution_contexts;
    std::unordered_set< onika::parallel::ParallelExecutionContext* > m_allocated_parallel_execution_contexts;
        
    // Operator protection after compilation
    bool m_compiled = false;

    // profiling
    bool m_profiling = true;
    
    // allow OpenMP task creation and shared parallel operation stream queues across several operators when activated at a batch level
    bool m_omp_task_mode = false;
    
    // allow gpu execution for this particular instance
    bool m_gpu_execution_allowed = true;
    
    static ProfilingFunctionSet s_profiling_functions;
    static TimeStampT s_profiling_timestamp_ref;
    
    // GPU context
    static std::shared_ptr<onika::cuda::CudaContext> s_global_cuda_ctx;
    
    static bool s_global_profiling;
    static bool s_global_mem_profiling;
    static bool s_debug_execution;
  };

  using OperatorDebugLogFilter = OperatorNode::OperatorDebugStreamHelper;
}



