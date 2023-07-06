#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_task.h>
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>

#include <memory>
#include <string>
#include <limits>
#include <omp.h>

namespace exanb
{

  // =====================================================================
  // ========================== OperatorBatchNode ========================
  // =====================================================================

 // a batch operator node contains a bunch of operators added sequentially
  // exposed slots of added nodes are also exposed to this batch node
  class OperatorBatchNode : public OperatorNode
  {
  public:

    void run() override final;
    void execute() override final;
    void generate_tasks() override final;
    bool is_terminal() const override final;
    LogStreamWrapper& pretty_print(LogStreamWrapper& out, int details, int indent, ProfilePrintParameters& ppp , ParallelValueStatsFunc pstat ) override final;
    void apply_graph( std::function<void(OperatorNode*)> f, bool prefix=false) override final;
    OperatorNodeFlavor in_flavor() const override final;
    OperatorNodeFlavor out_flavor() const override final;
    void compile() override final;
    void post_graph_build() override final;

    inline void set_condition_inverse(bool inv) { m_condition_inv = inv; assert(m_condition_slot==nullptr); }
    inline bool condition_inverse() const { return m_condition_inv; }
    inline void set_condition(const std::string& s) { m_condition=s; }
    inline const std::string& condition() const { return m_condition; }
    
    inline bool is_conditional() const { return ! m_condition.empty(); }
    inline bool eval_condition()
    {
      if( m_condition_slot != nullptr )
      {
        m_condition_slot->initialize_resource_pointer();
        assert( m_condition_slot->resource()->memory_ptr()!=nullptr );
        assert( m_condition_slot->has_value() );        
        return m_condition_slot->value_as_bool() ^ m_condition_inv;
      }
      else
      {
        assert( ! is_conditional() );
        return true;
      }
    }
    
    // loop execution of batch
    inline void set_looping(bool mlc) { m_looping = mlc; }
    bool is_looping() const override final;
    void finalize() override final;
    
    // concurrent execution. cannot be used if operator task_parallelism() returns true 
    inline void set_parallel_execution(bool p) { m_parallel_execution=p; }
    inline bool parallel_execution() const { return m_parallel_execution; }
    inline void set_parallel_cores(const std::vector<double>& partition) { m_parallel_cores = partition; }    
        
    void push_back( std::shared_ptr<OperatorNode> op );
    std::shared_ptr<OperatorNode> at(size_t i) const;
    
    bool is_trivial_encapsulation() const ;

    static std::shared_ptr<OperatorNode> make_operator_batch(const YAML::Node& batch_node, const OperatorNodeFlavor& flavor);

  protected:
    double collect_execution_time() override final;

  private:
    void set_rebind_name( const std::string& k , const std::string& otherk );
    const std::string& rebind_name( const std::string& k ) const;
    const std::string& rebind_reverse_name( const std::string& k ) const;

    // executes sub operators sequentially
    void execute_sequential_ops();
  
    // inner operators
    std::vector< std::shared_ptr<OperatorNode> > m_ops;
    
    // account for successive iterations when collecting execution times
    double m_previous_iteration_time = 0.0;
    
    // batch level slot renaming
    std::map< std::string, std::string > m_rebind;
    std::map< std::string, std::string > m_rebind_reverse;

    // conditional execution
    std::string m_condition;
    std::shared_ptr<OperatorSlotBase> m_condition_slot;
    
    // OpenMP nested parallelism control
    std::vector<double> m_parallel_cores;
    bool m_parallel_execution = false;
    
    // condition inverter
    bool m_condition_inv = false;

    // loop execution flag
    bool m_looping = false;
  };





  // ********************************************************************************
  // ***************************** Implementation ***********************************
  // ********************************************************************************

  bool OperatorBatchNode::is_looping() const
  {
    return m_looping;
  }

  void OperatorBatchNode::finalize()
  {
    for(auto& f : m_ops)
    {
      f->finalize();
    }    
  }

  std::shared_ptr<OperatorNode> OperatorBatchNode::at(size_t i) const
  {
    return m_ops[i];
  }

  bool OperatorBatchNode::is_trivial_encapsulation() const
  {
    return m_ops.size()==1 && !is_looping() && !is_conditional() && !task_parallelism();
  }

  void OperatorBatchNode::set_rebind_name( const std::string& k , const std::string& otherk )
  {
    m_rebind[ k ] = otherk;
    m_rebind_reverse[ otherk ] = k;
  }

  const std::string& OperatorBatchNode::rebind_name( const std::string& k ) const
  {
    auto it =  m_rebind.find( k );
    if( it ==  m_rebind.end() ) { return k; }
    else { return it->second; }
  }

  const std::string& OperatorBatchNode::rebind_reverse_name( const std::string& k ) const
  {
    auto it =  m_rebind_reverse.find( k );
    if( it ==  m_rebind_reverse.end() ) { return k; }
    else { return it->second; }
  }

  bool OperatorBatchNode::is_terminal() const
  {
    return false;
  }

  /*! in the case of a batch node, the flavor is computed
   * according to what would be the input slots after connections are made
   */
  OperatorNodeFlavor OperatorBatchNode::in_flavor() const
  {
    OperatorNodeFlavor f;
    for(auto op_it=m_ops.begin(); op_it!=m_ops.end(); ++op_it )
    {
      auto op = *op_it;
      // std::set<std::string> no_input_exposure;
      for(auto key_value : op->in_slots() )
      {
        const std::string k = key_value.first;
        const std::string batch_k = rebind_name( k );
        OperatorSlotBase* s = key_value.second;
        if( s->is_input_connectable() )
        {
          bool connection_found = false;
          for(auto it=std::make_reverse_iterator(op_it) ; !connection_found && it!=m_ops.rend() ; ++it)
          {
            assert( *it != *op_it );
            if( (*it)->out_slot(k) != nullptr )
            {
              connection_found = (*it)->out_slot(k)->is_output_connectable();
            }
          }
          if( !connection_found && f.find(batch_k)==f.end() )
          {
            f[ batch_k ] = s->value_type();
          }
          // if( connection_found ) { no_input_exposure.insert(k); }
        }
        // else { no_input_exposure.insert(k); }
      }
      /*
      auto opinflv = op->in_flavor();
      for(const auto& key_value : opinflv )
      {
        const std::string& k = key_value.first;
        const std::string& batch_k = rebind_name( k );
        const std::string& value_type = key_value.second;
        if( no_input_exposure.find(k) == no_input_exposure.end() && f.find(batch_k)==f.end() )
        {
          f[ batch_k ] = value_type;
        }
      }
      */
    }
        
    if( ! m_condition.empty() ) { f[m_condition] = typeid(bool).name(); }
    return f;
  }

  /*! in the case of a batch node, the flavor is computed
   * according to what would be the output slots after connections are made
   */
  OperatorNodeFlavor OperatorBatchNode::out_flavor() const
  {
    OperatorNodeFlavor f;
    for(auto it=m_ops.rbegin() ; it!=m_ops.rend() ; ++it)
    {
      OperatorNodeFlavor of = (*it)->out_flavor();
      for(const auto& key_value : of )
      {
        const std::string k = key_value.first;
        const std::string batch_k = rebind_name( k );
        const std::string value_type = key_value.second;
        if( f.find(batch_k)==f.end() )
        {
          f[ batch_k ] = value_type;
        }
        else
        {
          if( f[ batch_k ] != value_type )
          {
            lerr << "Warning, conflicting type for batch slot '"<<batch_k<<"'"<<std::endl
                 << "  reject type "<<pretty_short_type(value_type)<<std::endl
                 << "  keep type "<<pretty_short_type(f[batch_k])<<std::endl;
          }
        }
      }
    }
    return f;
  }

  // call this once, when no more push_back has to be done
  void OperatorBatchNode::compile()
  {
    assert( ! compiled() );

    for(auto op_it=m_ops.begin(); op_it!=m_ops.end(); ++op_it )
    {
      (*op_it)->set_parent( this );
    }

    /*** slot connection ***/
    for(auto op_it=m_ops.begin(); op_it!=m_ops.end(); ++op_it )
    {
      auto op = *op_it;

      // connect operator slots
      for(auto key_value : op->in_slots() )
      {
        const std::string k = key_value.first;
        const std::string batch_k = rebind_name( k );
        OperatorSlotBase* s = key_value.second;

        if( s->is_input_connectable() )
        {
          bool connection_found = false;
          for(auto it=make_reverse_iterator(op_it) ; !connection_found && it!=m_ops.rend() ; ++it)
          {
            assert( *it != *op_it );
            if( (*it)->out_slot(k) != nullptr )
            {
              OperatorSlotBase::connect( (*it)->out_slot(k) , s );
              connection_found = true;
            }
          }

          if( ! connection_found )
          {
            OperatorSlotBase* bs = in_slot(batch_k);
            if( bs == nullptr )
            {
              // this allows slot to be a pipe (in/out) but registered only on the input side
              // this is valid because shared_ptr has been referenced with register_managed_slot inside clone/make_operator_slot method
              bs = s->new_instance( this, batch_k, INPUT ).get(); 
              bs->set_inout();
              bs->set_resource( std::make_shared<OperatorSlotResource>(nullptr) );
              assert( bs->resource()->is_null() );
            }
            OperatorSlotBase::connect( bs , s );
          }
        }
      }
    }
    /*** end of slot connection ***/


    //************** condition slot creation ******************
    if( is_conditional() )
    {
      assert( m_condition_slot == nullptr );
      OperatorSlotBase* cs = in_slot(m_condition);
      if( cs != nullptr )
      {
        ldbg << "condition slot "<<m_condition<< " already exists, reusing it." <<std::endl;
        assert( cs->owner() == this );
        for( auto& p : managed_slots() ) { if(p.get()==cs) { m_condition_slot = p; } }
      }
      else
      {
        ldbg << "create condition slot "<<m_condition<< std::endl;
        m_condition_slot = make_operator_slot<bool>( this , m_condition, INPUT );
      }
      m_condition_slot->set_conditional_input( true );
      
      assert( m_condition_slot != nullptr );
      assert( m_condition_slot->value_type() == typeid(bool).name() );
      m_condition_slot->set_required( true );
      assert( m_condition_slot.get() == in_slot(m_condition) );
      assert( slots().find(m_condition_slot.get()) != slots().end() );
    }
    //*********************************************************


    //************** expose all output slots at batch level ****************
    for(auto it=m_ops.rbegin() ; it!=m_ops.rend() ; ++it)
    {
      for(auto key_value : (*it)->out_slots() )
      {
        const std::string k = key_value.first;
        const std::string batch_k = rebind_name( k );
        OperatorSlotBase* s = key_value.second;
        OperatorSlotBase* bs = out_slot(batch_k);
        if( bs == nullptr && s->is_output_connectable() )
        {
          // this allows slot to be a pipe (in/out) but registered only on the input side
          bs = s->new_instance( this, batch_k, OUTPUT ).get();
          bs->set_inout();
          bs->set_resource( std::make_shared<OperatorSlotResource>(nullptr) );
          assert( bs->resource()->is_null() );
          OperatorSlotBase::connect( s , bs );
        }
      }
    }
    // ***************************************************************

    // sanity check
    if( m_condition_slot != nullptr )
    {
      assert( slots().find(m_condition_slot.get()) != slots().end() );
    }

    if( m_looping )
    {
      for(auto& os : out_slots())
      {
        OperatorSlotBase* is = in_slot( os.first );
        if( is==nullptr )
        {
          ldbg << os.second->pathname() <<" not loop connected"<<std::endl;
        }
        if( is != nullptr && is->is_input_connectable() )
        {
          if( is->value_type() == os.second->value_type() )
          {
            ldbg << "loop connect "<<os.second->pathname()<<" to "<<is->pathname()<<std::endl;
            if( os.second->loop_output() != nullptr )
            {
              lerr<<"loop_output already connected"<<std::endl;
              std::abort();
            }
            if( is->loop_input() != nullptr )
            {
              lerr<<"loop_input already connected"<<std::endl;
              std::abort();
            }
            os.second->set_loop_output( is );
            is->set_loop_input( os.second );
          }
          else
          {
            lerr << "CANNOT loop connect "<<os.second->pathname()<<" to "<<is->pathname()<< " (types differ)"<<std::endl;
          }
        }
      }
    }

    this->OperatorNode::compile();    
    //ldbg << "OperatorBatchNode::compile end" << std::endl;
  }

  /*!
   * Adds an operator to this batch, at the end of operator list
   */
  void OperatorBatchNode::push_back( std::shared_ptr<OperatorNode> op )
  {
    assert( ! compiled() );  
    if( op->name() == "nop" ) return ; // TODO: this is ugly, has to be changed
    m_ops.push_back( op );
  }

  /*!
   * main execution function, in the case of a batch, evaluates condition (if any)
   * prior to execution
   */
  void OperatorBatchNode::run()
  {
    // FIXME: something to do here for task dependencies ...
    // case 0 : task_parallelism() => false, we don't need any dependency handling
    // case 1 : task_parallelism() => true, parent()->task_parallelism() => false,
    //          tasking not started, we'll start tasking in generate_tasks method, don't need to handle dependency
    //          because previous operators ran sequentially
    // case 2 : if we're already in tasking mode, we can safely use a task with dependency
    if( eval_condition() )
    {
      this->OperatorNode::run();
    }    
  }

  void OperatorBatchNode::execute_sequential_ops()
  {
    for(const auto& f : m_ops)
    {
//      std::cout << "run "<< f->name() << std::endl;
      f->run(); 
    }
  }

  void OperatorBatchNode::generate_tasks()
  {
    if( is_looping() || is_conditional() )
    {
      lerr << "OperatorBatchNode::generate_tasks : batch loop and condition not allowed in tasking mode";
      std::abort();
    }
  
    if( parallel_execution() )
    {
      lerr << "Internal error: OperatorBatchNode::generate_tasks : cannot use both parallel and parallel_task execution mode";
      std::abort();
    }

    bool start_tasking = task_parallelism();
    if( parent() != nullptr )
    {
      start_tasking = start_tasking && !parent()->task_parallelism();
    }

    if( start_tasking ) // start a tasking region
    {
#     pragma omp parallel
      {
#       pragma omp single
        {
          //ONIKA_DBG_MESG_LOCK { lout<< name() << " : start ptask scheduler" << std::endl << std::flush; }
          ptask_queue().start();

#         pragma omp taskgroup
          {
            //ONIKA_DBG_MESG_LOCK { lout<< name() << " : start tasking mode (" << m_ops.size() <<" ops)"<<std::endl << std::flush; }  
            execute_sequential_ops();
            //ONIKA_DBG_MESG_LOCK { lout<< name() << " : finished spwan tasks" << std::endl << std::flush; }
          } // --- end of task group ---
          
          // when primary tasks are finished, we can request task scheduler to terminate after all enqueued tasks are processed
          //ONIKA_DBG_MESG_LOCK { lout<< name() << " : ptq stop" << std::endl << std::flush; }
          ptask_queue().flush();           
          //ONIKA_DBG_MESG_LOCK { lout<< name() << " : ptq stop" << std::endl << std::flush; }
          ptask_queue().stop();
          //ONIKA_DBG_MESG_LOCK { lout<< name() << " : ptq wait_all" << std::endl << std::flush; }
          ptask_queue().wait_all();
        } // --- end of single ---

        // synchronize with OperatorTaskScheduler 
/*
#       pragma omp single nowait
        {
          ONIKA_DBG_MESG_LOCK { lout<< name() << " : end tasking mode" << std::endl << std::flush; }
        }
*/
      }
    }
    else
    {
      execute_sequential_ops();
    }
  }

  // execute this batch
  void OperatorBatchNode::execute()
  {
    size_t iteration_count = 0;
    bool continue_loop = true;

    if( task_parallelism() )
    {
      lerr << "Internal error: OperatorBatchNode::execute : cannot use both parallel and parallel_task execution mode";
      std::abort();
    }

    while( continue_loop )
    {
      /*
      parallel execution means that a parallel section will be started
      before operator is execute method is called.
      this can achieve two things :
        1. concurrent execution of sub operators with different thread subsets assigned to each through OpenMP nested parallelism feature
        2. allow sub operators not to create a parallel section, but rather create tasks.
           subsequent operators will be able to provide tasks for execution in a common task group
      */
      if( parallel_execution() )
      {      
        int num_threads = omp_get_max_threads();
        int n_ops = m_ops.size();
        if( n_ops > num_threads ) { lerr << "Warning: more parallel operators than available threads" << std::endl; }
        
        double thread_normalized_sum = 0.0;
        int pccount = m_parallel_cores.size();
        for(int i=0;i<n_ops;i++)
        {
          if(i<pccount) { thread_normalized_sum += m_parallel_cores[i]; }
          else { thread_normalized_sum += 1.0; }
        }

        int allocated_threads[n_ops+1];
        double ratio_sum = 0.0;
        for(int i=0;i<n_ops;i++)
        {
          double core_ratio = 1.0;
          if(i<pccount) { core_ratio = m_parallel_cores[i]; }
          allocated_threads[i] = static_cast<int>( std::round( ( ratio_sum * num_threads ) / thread_normalized_sum ) );
          ratio_sum += core_ratio;
        }
        allocated_threads[n_ops] = num_threads;

#       pragma omp parallel num_threads(n_ops)
        {
          int task_index = omp_get_thread_num();
          int task_threads = allocated_threads[task_index+1] - allocated_threads[task_index];
          if( task_threads < 1 ) { task_threads = 1; }
          omp_set_num_threads( task_threads );
          m_ops[task_index]->run();
        }
      }
      else
      {
        execute_sequential_ops();
      }
      
      ++ iteration_count;
      
      if( is_looping() ) { continue_loop = eval_condition(); }
      else { continue_loop = false; }
      
      if( continue_loop )
      {
        m_previous_iteration_time += this->collect_execution_time();
      }
    }

#   ifndef NDEBUG
    if( is_looping() )
    {
#     pragma omp critical(OperatorBatchNode_loop_dbg_message)
      {
        ldbg << format_string("%*s",(depth()-1)*2,"") << name() << " (executed "<< iteration_count << " iterations)" << std::endl ;
      }
    }
#   endif
  }

  double OperatorBatchNode::collect_execution_time()
  {
#   ifdef ONIKA_HAVE_OPENMP_TOOLS
    bool do_profiling = global_profiling() && profiling();
    if( do_profiling )
    {
      auto t = m_task_exec_time_accum.exchange(0) / 1000000.0;
      for(auto op : m_ops) { t += op->collect_execution_time(); }
      if( t > 0.0 ) { m_exec_times.push_back( t ); }
      t += m_previous_iteration_time;
      m_previous_iteration_time = 0.0;
      return t;
    }
    else
#   endif
    {
      return 0.0;
    }
  }

  LogStreamWrapper& OperatorBatchNode::pretty_print(LogStreamWrapper& out, int details, int indent, ProfilePrintParameters& ppp, ParallelValueStatsFunc pstat )
  {
    const double inner_loop_backup = ppp.m_inner_loop_time;    
    this->OperatorNode::pretty_print(out,details,indent,ppp,pstat);     
    for(auto op : m_ops)
    {
      op->pretty_print(out,details,indent+2,ppp,pstat);
    }
    ppp.m_inner_loop_time = inner_loop_backup;
    return out;
  }

  void OperatorBatchNode::apply_graph( std::function<void(OperatorNode*)> f, bool prefix)
  {
    if( !prefix ) this->OperatorNode::apply_graph( f , prefix );
    for(auto op : m_ops) { op->apply_graph( f , prefix ); }    
    if( prefix ) this->OperatorNode::apply_graph( f , prefix );
  }

  std::shared_ptr<OperatorNode> OperatorBatchNode::make_operator_batch(const YAML::Node& _batch_node, const OperatorNodeFlavor& flavor)
  {
    YAML::Node batch_node = YAML::Clone(_batch_node);
    
    std::string tmp_operator_name = "<anonymous>";
    auto op_name_it = flavor.find("__operator_name__");
    if( op_name_it != flavor.end() ) { tmp_operator_name = op_name_it->second; }
  
    std::shared_ptr<OperatorBatchNode> batch = std::make_shared<OperatorBatchNode>();
    
    bool batch_profiling_forced = false;
    bool batch_profiling = true;
    
    YAML::Node oplist_node;
    // if we have just a list, it means it start directly with a list of sub operators (what you would normally find in the body: key)
    if( batch_node.IsSequence() )
    {
      // ldbg << "list only batch" << std::endl;
      oplist_node = batch_node;
    }
    // if it's a string, it's an alias
    else if( batch_node.IsScalar() )
    {
      // ldbg << "batch is an alias to " << batch_node.as<std::string>() << std::endl;
      return OperatorNodeFactory::instance()->make_operator( batch_node.as<std::string>() , YAML::Node() , flavor );
    }
    else
    {
      if( ! batch_node.IsMap() )
      {
        lerr << "YAML node for batch operator is neither a list neither a map."<<std::endl;
        std::abort();
      }
      
      if( ! static_cast<bool>( batch_node["body"] ) )
      {
        lerr << "YAML node for batch "<<tmp_operator_name<<" is not a list but does not contain a 'body' entry"<<std::endl;
        return nullptr;
      }
      
      // transform operator tree to unroll a loop batch
      if( batch_node["unroll"] && batch_node["loop"] && batch_node["body"] )
      {
        bool loop = batch_node["loop"].as<bool>();
        int ul = batch_node["unroll"].as<int>();
        if( loop && ul>1 )
        {
          ::exanb::ldbg << "unroll "<<tmp_operator_name<<" x"<<ul<<std::endl;
          YAML::Node sub_batch = YAML::Clone(batch_node);
          sub_batch.remove("unroll");
          sub_batch.remove("loop");
          YAML::Node unrolled_body( YAML::NodeType::Sequence );
          for(int i=0;i<ul;i++)
          {
            YAML::Node unrolled_instance = YAML::Clone(sub_batch);
            std::ostringstream oss; oss<<tmp_operator_name<<"$"<<i;
            std::string opname = oss.str();
            //for(auto& c:opname) if(c=='.') c='_';
            unrolled_instance["name"] = opname;
            YAML::Node batch_encap( YAML::NodeType::Map );
            batch_encap[ opname ] = unrolled_instance;
            unrolled_body.push_back( batch_encap );
          }
          batch_node["body"] = unrolled_body;
          batch_node.remove( "rebind" );
          // ldbg << "unrolled batch :" << std::endl;
          // dump_node_to_stream( ldbg , batch_node );
        }
      }
      
      // handle slot exported with alternative names
      if( batch_node["rebind"] )
      {
        if( ! batch_node["rebind"].IsMap() )
        {
          lerr << "rebind property must be a map"<<std::endl;
          return nullptr;
        }
        for(auto kv : batch_node["rebind"])
        {
          // ldbg << "rebind "<< kv.first.as<std::string>() << " to "<<kv.second.as<std::string>()<<std::endl;
          std::string internal_name = kv.first.as<std::string>();
          std::string external_name = kv.second.as<std::string>();
          batch->set_rebind_name( internal_name, external_name );
        }
        
      }
      
      // add a condition slot to drive execution of this batch
      if( batch_node["condition"] )
      {
        std::string s = batch_node["condition"].as<std::string>();
        std::istringstream iss(s);
        std::string s1, s2;
        iss >> s1 >> s2 ;
        if( s1 == "not" )
        {
          batch->set_condition(s2);
          batch->set_condition_inverse(true);
          if( OperatorNodeFactory::debug_verbose_level() >= 2 ) { ::exanb::ldbg << "condition: ! "<< s2 << std::endl; }
        }
        else
        {
          batch->set_condition(s1);
          batch->set_condition_inverse(false);
          if( OperatorNodeFactory::debug_verbose_level() >= 2 ) { ::exanb::ldbg << "condition: "<< s1 << std::endl; }
        }
      }

      if( batch_node["name"] )
      {
        // ldbg << "set name to "<<batch_node["name"].as<std::string>()<< std::endl;
        batch->set_name( batch_node["name"].as<std::string>() );
      }
      
      if( batch_node["loop"] )
      {
        batch->set_looping( batch_node["loop"].as<bool>() );
      }        
      
      if( batch_node["profiling"] )
      {
        batch_profiling_forced = true;
        batch_profiling = batch_node["profiling"].as<bool>();
      }
  
      if( batch_node["parallel"] )
      {
        std::cout << "parallel batch "<<tmp_operator_name<<std::endl;
        batch->set_parallel_execution( batch_node["parallel"].as<bool>() );
        if( batch->parallel_execution() )
        {
          if( batch_node["cores"] )
          {
            batch->set_parallel_cores( batch_node["cores"].as< std::vector<double> >() );
          }
        }
      }
      else if( batch_node["parallel_tasks"] )
      {
        bool pt = batch_node["parallel_tasks"].as<bool>();
        //std::cout << tmp_operator_name<< " : parallel_tasks="<<std::boolalpha<<pt<<std::endl;
        batch->set_task_parallelism( pt );
      }
      
      oplist_node = batch_node["body"];
    }
    
//    lout<<"========== building "<< tmp_operator_name <<" (" << batch->name() <<") ==========\n";

    for(YAML::Node op : oplist_node)
    {
      YAML::Node params;
      std::string opname;
      if( op.IsMap() ) // if the node has the form '- compute_op: <parameters map> '
      {
        if( op.size() != 1 )
        {
          lerr<<"operator has the form compute_op: <parameters map>, but its map size is "<<op.size()<<std::endl;
          lerr<<"node content was :"<<std::endl;
          dump_node_to_stream( lerr , op );
          lerr<<std::endl;
          std::abort();
        }
        opname = op.begin()->first.as<std::string>();
        params = op.begin()->second;
      }
      else if( op.IsScalar() ) // if it has the form '- compute_op'
      {
        opname = op.as<std::string>();
      }
      else
      {
        lerr<<"bad YAML type for operator"<<std::endl;
        std::abort();
      }

      // flavor is inerited from encapsulating batch (if any) and progressively overwritten by current batch's flavor
            
      OperatorNodeFlavor f;
      for(const auto& p : flavor)
      {
        if( batch->rebind_name(p.first) == p.first )
        {
          f[batch->rebind_reverse_name(p.first)] = p.second;
        }
      }      
      for(const auto& p : batch->flavor() )
      {
        f[batch->rebind_reverse_name(p.first)] = p.second;
      }

      if( OperatorNodeFactory::debug_verbose_level() >= 3 )
      {
        ::exanb::ldbg << "batch "<<tmp_operator_name<<" add operator "<<opname<<" with flavor :" << std::endl;
        for(const auto& p : f)
        {
          if( p.first.find("__") != 0 )
          {
            ::exanb::ldbg << '\t' << p.first << " -> " << pretty_short_type(p.second) << std::endl;
          }
        }
      }

      std::shared_ptr<OperatorNode> new_op = nullptr;
      try
      {
        new_op = OperatorNodeFactory::instance()->make_operator( opname, params, f );
      }
      catch( const OperatorCreationException& e )
      {
        lerr<<"Unable to add operator '"<<opname<<"' to batch"<<std::endl;
        lerr<< str_indent( e.what() , 2 , ' ' , "| " );
        std::abort();
      }
      if( new_op == nullptr )
      {
        lerr<<"Unable to add operator '"<<opname<<"' to batch (unidentified error)"<<std::endl;
        std::abort();
      }
      
      batch->push_back( new_op );
    }

/*
    lout<<"---------------------------------\n";
    batch->OperatorNode::pretty_print(lout, 0 );
    lout<<"=================================\n\n";
*/

    std::shared_ptr<OperatorNode> final_op = batch;
    if( batch->is_trivial_encapsulation() )
    {
      ::exanb::ldbg  << "Simplify "<<tmp_operator_name << std::endl;
      std::shared_ptr<OperatorNode> op = batch->at(0);
      for(auto key_value : op->in_slots() )
      {
        std::string k = key_value.first;
        std::string batch_k = batch->rebind_name( k );
        if( k != batch_k )
        {
          op->in_slot_rename( k , batch_k );
          assert( op->in_slot(batch_k) == key_value.second );
        }
      }
      for(auto key_value : op->out_slots() )
      {
        std::string k = key_value.first;
        std::string batch_k = batch->rebind_name( k );
        if( k != batch_k )
        {
          op->out_slot_rename( k , batch_k );
          assert( op->out_slot(batch_k) == key_value.second );
        }
      }
      final_op = op;
    }
    
    batch = nullptr;

/*
    lout<<"========== final "<< tmp_operator_name <<" (" << final_op->name() <<") ==========\n";
    final_op->pretty_print(lout, 0 );
    lout<<"=================================\n\n";
*/

    if( batch_profiling_forced )
    {
      final_op->apply_graph( [p=batch_profiling](OperatorNode* o){ o->set_profiling(p); } );
    }

    return final_op;
  }

  // optimization opportunities here
  void OperatorBatchNode::post_graph_build()
  {
    // detect operators that produce ouputs that will never be consumed
    std::set< std::shared_ptr<OperatorNode> > to_remove;
    for(auto op : m_ops)
    {
      if( op->is_terminal() )
      {
        if( ! op->is_sink() )
        {
          std::set<const OperatorSlotBase*> ors;
          op->outflow_reachable_slots(ors);
          bool has_outflow_sink = false;
          std::set<const OperatorNode*> reachable_sink_operators;
          for(auto os:ors)
          {
            if( ( os->owner()->is_sink() && os->owner()->is_terminal() ) || os->is_conditional_input() )
            {
              reachable_sink_operators.insert( os->owner() );
              has_outflow_sink = true;
            }
          }

          if( ! has_outflow_sink )
          {
            to_remove.insert(op);
          }
        }
      }
    }
    
    for(auto op:to_remove)
    {
      lout << "*SUPPRESS* "<<op->pathname()<<std::endl;
      for(auto s:op->slots()) s->remove_connections();
      auto it = std::find( m_ops.begin(), m_ops.end() , op );
      m_ops.erase( it );
    }
    
    this->OperatorNode::post_graph_build();
  }

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "batch", OperatorBatchNode::make_operator_batch );
  }

}

