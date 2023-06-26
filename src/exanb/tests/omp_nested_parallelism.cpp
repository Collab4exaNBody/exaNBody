#include <iostream>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include <thread>
#include <set>

#define TEST_ASSERT(cond) do{ if(!(cond)) { std::cerr<<"assertion '"<< #cond <<"' failed"<<std::endl;  std::abort(); } }while(false)

std::set<std::thread::id> involved_threads;

void report_task_threads(int task)
{
      int tid = omp_get_thread_num();
#     pragma omp critical
      {
        involved_threads.insert( std::this_thread::get_id() );
        std::cout<<"thread #"<<tid<<" (id=" << std::this_thread::get_id() <<") in task #"<<task<<std::endl;
      }    
}


void worksharing_task(int i)
{
  int tid = omp_get_thread_num();
# pragma omp critical
  {
    std::cout<<"thread #"<<tid<<" (id=" << std::this_thread::get_id() <<") workshared element #"<<i<<std::endl << std::flush;
  }
}

void standalone_task(int task)
{
  int tid = omp_get_thread_num();
  // int max_threads = omp_get_max_threads();
# pragma omp critical
  {
    std::cout<<"thread #"<<tid<<" (id=" << std::this_thread::get_id() <<") standalone task #"<<task /* <<" MT="<<max_threads */ << std::endl << std::flush;
  }
}


std::set<std::thread::id> par_in_task_threads;
void parallel_region_in_task(int task)
{
  int tid = omp_get_thread_num();
  int max_threads = omp_get_max_threads();
# pragma omp parallel
  {
    int inner_tid = omp_get_thread_num();
#   pragma omp critical
    {
      par_in_task_threads.insert( std::this_thread::get_id() );
      std::cout<<"par_in_task tid="<<tid<<", id=" << std::this_thread::get_id() <<", task #"<<task<<", in_tid="<<inner_tid<<", MT="<<max_threads<< std::endl<<std::flush;
    }
  }

}


int main(int argc,char*argv[])
{
  unsigned int nsubregions = 2;
  if(argc>=2)
  {
    nsubregions = std::atoi(argv[1]);
  }
  TEST_ASSERT( nsubregions>=1 );

  std::cout << "-------- test nested parallelism -------" << std::endl;

  int n_threads = 0;
  int threads_idx_sum = 0;
# pragma omp parallel
  {
    int tid = omp_get_thread_num();
#   pragma omp critical
    {
      int nt = omp_get_num_threads();
      TEST_ASSERT( n_threads==0 || n_threads==nt );
      n_threads = nt;
    }
#   pragma omp atomic
    threads_idx_sum += tid;
  }
  TEST_ASSERT( threads_idx_sum == (n_threads*(n_threads-1))/2 );	

  omp_set_nested(1);
  int is_nested = omp_get_nested();
  TEST_ASSERT( is_nested );

  int task_threads = n_threads / nsubregions;
  TEST_ASSERT( task_threads>=1 );
   
# pragma omp parallel num_threads(nsubregions)
  {
    int taskindex = omp_get_thread_num();
#   pragma omp critical
    {
      std::cout<<"main thread #"<<taskindex<<" (id=" << std::this_thread::get_id() << ") starts parallel task with "<<task_threads <<" threads"<<  std::endl;
      involved_threads.insert( std::this_thread::get_id() );
    }
    
    omp_set_num_threads(task_threads); // this replaces num_threads(task_threads) on the following line
    // allowing for sub region unaware code to execute properly
#   pragma omp parallel //num_threads(task_threads)
    {
      report_task_threads(taskindex);
    }
  }

  std::cout <<n_threads<<" thread(s)"<<", nested="<<omp_get_nested()<<", tasks="<<nsubregions<<", threads per task="<<task_threads<<", total threads="<<involved_threads.size() <<std::endl;
  
  TEST_ASSERT( involved_threads.size() == nsubregions*task_threads );
  
  
  
  std::cout << "-------- test tasks -------" << std::endl;

# pragma omp parallel
  {
#   pragma omp single nowait
    {
      for(int i=0;i<10;i++)
      {
#       pragma omp task
        standalone_task(i);
      }
    }
    
#   pragma omp for nowait
    for(int i=0;i<10;i++)
    {
      worksharing_task(i);
    }
  }

  std::cout << "-------- test with set_num_threads(1) -------" << std::endl;
# pragma omp parallel num_threads(1)
  {
    omp_set_num_threads(1);
#   pragma omp parallel
    {
      std::cout << "thread "<<omp_get_thread_num()<<std::endl;
    }
  }

  std::cout << "-------- test without set_num_threads -------" << std::endl;
# pragma omp parallel
  {
    std::cout << "thread "<<omp_get_thread_num()<<std::endl;
  }
  

  std::cout << "-------- test parallel region in task -------" << std::endl;
# pragma omp parallel //num_threads(1)
  {
  
/*
    // sequential task generation
#   pragma omp single nowait
    {
      for(int i=0;i<4;i++)
      {
#       pragma omp task
          parallel_region_in_task(i);
      }
    }
*/

    // parallel task generation
#   pragma omp task
    parallel_region_in_task( omp_get_thread_num() );

#   pragma omp critical
    std::cout << "thread #"<<omp_get_thread_num()<<" ready to execute tasks"<<std::endl<<std::flush;

#   pragma omp taskwait

#   pragma omp barrier

#   pragma omp critical
    std::cout << "thread #"<<omp_get_thread_num()<<" done"<<std::endl<<std::flush;
  }
  std::cout << par_in_task_threads.size() << " system threads involved"<<std::endl<<std::endl;



  std::cout << "-------- test set_num_threads persistence -------" << std::endl;
  int max_threads_a = -1;
  int max_threads_b = -1;
  
# pragma omp parallel num_threads(1)
  {
    omp_set_num_threads(2);
#   pragma omp parallel
    {
#     pragma omp single
      max_threads_a = omp_get_max_threads();
    }
  }

# pragma omp parallel num_threads(1)
  {
#   pragma omp parallel
    {
#     pragma omp single
      max_threads_b = omp_get_max_threads();
    }
  }
  std::cout << "nested parallel sections : first time "<< max_threads_a << ", second time "<<max_threads_b<<std::endl;

  max_threads_a = -1;
  max_threads_b = -1;
  omp_set_num_threads(2);
# pragma omp parallel
  {
#   pragma omp single
    max_threads_a = omp_get_max_threads();
  }
# pragma omp parallel
  {
#   pragma omp single
    max_threads_b = omp_get_max_threads();
  }
  std::cout << "non nested parallel sections : first time "<< max_threads_a << ", second time "<<max_threads_b<<std::endl;

  return 0;
}

