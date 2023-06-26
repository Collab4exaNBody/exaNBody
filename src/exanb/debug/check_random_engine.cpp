#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_random.h>

#include <mpi.h>

namespace exanb
{
  
  
  struct CheckRandomEngine : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi     , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( long     , cycles  , INPUT , 10 );
    ADD_SLOT( long     , samples , INPUT , 20 );

    inline void execute () override final
    {
      MPI_Comm comm = *mpi;

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&nprocs);
      
      int nsamples = *samples;
      int ncycles = *cycles;
      
      for(int c=0;c<ncycles;c++)
      {
        int tab[nsamples];
        for(int i=0;i<nsamples;i++)
        {
          tab[i] = -1;
        }
#       pragma omp parallel
        {
          auto& re = rand::random_engine();
          std::uniform_int_distribution<int> rint(1000,9999);

#         pragma omp for schedule(static)
          for(int i=0;i<nsamples;i++)
          {
            tab[i] = rint(re);
          }
        }
        
        lout << "cycle "<<c<<" :";
        for(int i=0;i<nsamples;i++)
        {
          lout <<' ' << tab[i];
        }
        lout << std::endl;
      }
    }

  };
  
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "check_random_engine", make_simple_operator< CheckRandomEngine > );
  }

}

