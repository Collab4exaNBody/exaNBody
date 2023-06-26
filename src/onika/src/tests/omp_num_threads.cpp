#include <iostream>
#include <cstdlib>
#include <cassert>
#include <omp.h>

int main(int argc,char*argv[])
{

# pragma omp parallel
  {
    int nt = omp_get_num_threads();
    _Pragma("omp single") std::cout<<"nt="<<nt<<"\n";
    int tid = omp_get_thread_num();
    _Pragma("omp critical(dbg_mesg)") std::cout<<"thread "<<tid<<" / "<<nt<<"\n";
  }

  return 0;
}

