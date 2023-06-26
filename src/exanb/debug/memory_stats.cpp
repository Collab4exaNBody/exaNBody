#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/operator_memory_stats.h>
#include <exanb/core/string_utils.h>

#include <mpi.h>
#include <vector>
#include <utility>
#include <algorithm>

#include <onika/memory/memory_usage.h>

// TODO: use malloc_info !!

namespace exanb
{
  
  
  // =====================================================================
  // ========================== Resource usage ===========================
  // =====================================================================

  class PrintRUsageOperator : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( bool     , rusage_stats, INPUT , true );
    ADD_SLOT( bool     , graph_slots, INPUT , true );
    ADD_SLOT( bool     , graph_res_mem, INPUT , false );
    ADD_SLOT( long     , musage_threshold   , 1024 );
    ADD_SLOT( long     , page_size , INPUT , 4096 );    
    
  public:
    
    inline bool is_sink() const override final { return true; } // not a suppressable operator
    
    inline void execute() override final
    {
      MPI_Comm comm = *mpi;
      int np=1;
      MPI_Comm_size(comm,&np);

      if( *rusage_stats )
      {
        onika::memory::MemoryResourceCounters memcounters;
        long stats_min[ memcounters.nb_counters() ];
        long stats_max[ memcounters.nb_counters() ];

        memcounters.read();
        MPI_Allreduce(memcounters.stats,stats_min,memcounters.nb_counters(),MPI_LONG,MPI_MIN,comm);
        MPI_Allreduce(memcounters.stats,stats_max,memcounters.nb_counters(),MPI_LONG,MPI_MAX,comm);
        MPI_Allreduce(MPI_IN_PLACE,memcounters.stats,memcounters.nb_counters(),MPI_LONG,MPI_SUM,comm);
        for(unsigned int i=0;i<memcounters.nb_counters();i++) memcounters.stats[i] /= np;

        lout<< "======================= Memory Usage Stats ========================="<<std::endl;
        lout<< "                                   average          min          max"<<std::endl;
        for(unsigned int i=0;i<memcounters.nb_counters();i++)
        {      
          lout<< format_string("%28s :%12s %12s %12s" , memcounters.labels[i] , large_integer_to_string(memcounters.stats[i]) , large_integer_to_string(stats_min[i]) , large_integer_to_string(stats_max[i]) ) << std::endl;
        }
        lout<< "===================================================================="<<std::endl<<std::endl;
      }
      
      if( *graph_slots )
      {
        OperatorNode* root = this;
        while( root->parent() != nullptr ) root = root->parent();

        lout<< "=============== Operator graph memory ==================="<<std::endl;
        print_operator_memory_stats( root , comm , *musage_threshold );
        lout<< "========================================================="<<std::endl<<std::endl;
      }

      if( *graph_res_mem )
      {
        using ResInfo = std::pair<std::string,ssize_t>;
        OperatorNode* root = this;
        while( root->parent() != nullptr ) root = root->parent();
        
        std::vector< ResInfo > resident_mem_inc;
        root->apply_graph(
          [&resident_mem_inc](OperatorNode* o)
          {
            resident_mem_inc.push_back( { o->name() , o->resident_memory_inc() } );
          }
        );
        std::sort( resident_mem_inc.begin() , resident_mem_inc.end() , [](const ResInfo& a, const ResInfo& b)->bool{return a.second > b.second;} );

        lout<< "=============== Resident memory leak ===================="<<std::endl;
        for(unsigned int i=0;i<10 && i<resident_mem_inc.size();i++)
        {
          lout << format_string("%35s : %12s",resident_mem_inc[i].first,large_integer_to_string(resident_mem_inc[i].second)) << std::endl;
        }
        lout<< "========================================================="<<std::endl<<std::endl;
      }

    }
  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "memory_stats", make_simple_operator< PrintRUsageOperator > );
  }

}

