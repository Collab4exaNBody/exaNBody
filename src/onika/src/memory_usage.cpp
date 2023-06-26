
#include <onika/memory/memory_usage.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

namespace onika
{
  namespace memory
  {

    const char * MemoryResourceCounters::labels[MemoryResourceCounters::N_COUNTERS] =
      {
      "total program size (Mb)",
      "resident set size (Mb)",
      "shared pages",
      "text (code) (Mb)",
      "libraries (Mb)",
      "data + stack (Mb)",
      "dirty pages",
      "page reclaims",
      "page faults",
      "swaps",
      "block input operations",
      "block output operations",
      "messages sent",
      "messages received",
      "signals received",
      "voluntary context switches",
      "involuntary context switches",
      "program break delta (Mb)"
      };

    void MemoryResourceCounters::read()
    {
      for(unsigned int i=0;i<N_COUNTERS;i++) stats[i] = 0;
    
      struct rusage ru;
      int rc = getrusage(RUSAGE_SELF, &ru);
      assert( rc == 0 );
      if( rc != 0 )
      {
        std::cerr << "getrusage failed" << std::endl;
      }
      std::ifstream statm("/proc/self/statm");
      if( statm )
      {
        statm >> stats[0] >> stats[1] >> stats[2] >> stats[3] >> stats[4] >> stats[5] >> stats[6];
        stats[0] = (stats[0]*page_size)/(1024*1024);
        stats[1] = (stats[1]*page_size)/(1024*1024);
        stats[3] = (stats[3]*page_size)/(1024*1024);
        stats[4] = (stats[4]*page_size)/(1024*1024);
        stats[5] = (stats[5]*page_size)/(1024*1024);
      }
      stats[7] = ru.ru_minflt;
      stats[8] = ru.ru_majflt;
      stats[9] = ru.ru_nswap;
      stats[10] = ru.ru_inblock;
      stats[11] = ru.ru_oublock;
      stats[12] = ru.ru_msgsnd;
      stats[13] = ru.ru_msgrcv;
      stats[14] = ru.ru_nsignals;
      stats[15] = ru.ru_nvcsw;
      stats[16] = ru.ru_nivcsw;

      static void * s_initial_brk = sbrk(0);
      stats[17] = ( (uint8_t*)sbrk(0) ) - ( (uint8_t*)s_initial_brk );
      stats[17] /= (1024*1024);

      //for(unsigned int i=0;i<N_COUNTERS;i++) stats_min[i] = stats_max[i] = stats[i];
    }

  }
}

