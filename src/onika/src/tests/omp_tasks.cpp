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
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <random>
#include <omp.h>

#ifdef ONIKA_HAVE_OPENMP_TOOLS
#include <omp-tools.h>
#endif

#include <onika/omp/task_detach.h>
#include <onika/omp/ompt_interface.h>
#include <onika/stream_utils.h>

#include <thread>
#include <set>
#include <chrono>
#include <atomic>


static std::ostream& my_app_context_printer( void* ctx , std::ostream& out )
{
  const char* appctx = reinterpret_cast<const char*>( ctx );
  if(appctx!=nullptr) out << appctx;
  else out << "null";
  return out;
}

static void my_task_start_callback( const onika::omp::OpenMPToolTaskTiming& evt )
{
  ONIKA_DBG_MESG_LOCK
  {
    std::cout << "task '"<< evt.tag <<"' start @ "<< evt.timepoint.count() << std::endl;
  }
}

static void my_task_stop_callback( const onika::omp::OpenMPToolTaskTiming& evt )
{
  ONIKA_DBG_MESG_LOCK
  {
    std::cout << "task '"<< evt.tag <<"' stop @ "<<  evt.timepoint.count() << std::endl;
  }
}


#define TEST_ASSERT(cond) do{ if(!(cond)) { std::cerr<<"assertion '"<< #cond <<"' failed at "<<__FILE__<<":"<<__LINE__<<std::endl;  std::abort(); } }while(false)

std::set<std::thread::id> involved_threads;

void my_task(int task, long delay)
{
  // in a task, max_threads and num_threads correspond to the enclosing parallel region current task belongs to
  int nt = omp_get_num_threads();
  int max_nt = omp_get_max_threads();
  int tid = omp_get_thread_num();
  auto sys_tid = std::this_thread::get_id();
  std::cout<<"thread #"<<tid<<" (id=" << sys_tid <<") start task #"<<task << " : nt="<<nt<<"/"<<max_nt <<std::endl;
  std::this_thread::sleep_for( std::chrono::milliseconds(delay) );
  std::cout<<"thread #"<<tid<<" (id=" << sys_tid <<") end task #"<<task<<std::endl;
}

int deps[1024];


void task_dependency(int second_task_creator_thread)
{

# pragma omp parallel
  {

#   pragma omp master
    {
      std::cout<<"master: tid="<<omp_get_thread_num() <<" nt="<<omp_get_num_threads()<<", max_nt="<<omp_get_max_threads()<<std::endl;
    }

#   pragma omp single
    {
      std::cout<<"single: tid="<<omp_get_thread_num() <<" nt="<<omp_get_num_threads()<<", max_nt="<<omp_get_max_threads()<<std::endl;
    }

    int tid = omp_get_thread_num();
    //int nt = omp_get_num_threads();

#   pragma omp barrier

    if( tid == 0 )
    {
      std::cout<<"thread "<<tid<<" creates task 0"<<std::endl;
#     pragma omp task depend(out:deps[0])
      {
        my_task(0 , 3000 );
      }
    }

    if( tid == second_task_creator_thread )
    {
      std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
      std::cout<<"thread "<<tid<<" creates task 1"<<std::endl;
#     pragma omp task depend(in:deps[0])
      { 
        my_task(1 , 1000 );
      }
    }

#   pragma omp taskwait

  }

}

void undeferred_task_dependency()
{
# pragma omp parallel
  {
#   pragma omp single nowait
    {
      std::cout << "thread #"<<omp_get_thread_num()<<" generates tasks"<<std::endl;
#     pragma omp task depend(out:deps[0])
      {
        std::cout<<"dep-out task started on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
        std::cout<<"dep-out task end"<<std::endl;
      }

#     pragma omp task depend(in:deps[0]) if(0)
      {
        std::cout<<"dep-in task executes on thread #"<<omp_get_thread_num()<<std::endl;
      }
      std::cout << "thread #"<<omp_get_thread_num()<<" finished task spawn"<<std::endl;
    }
#   pragma omp taskwait
  }
}


void undeferred_task_inner_sibling_dependency()
{
# pragma omp parallel
  {
#   pragma omp single nowait
    {
      std::cout << "thread #"<<omp_get_thread_num()<<" generates tasks"<<std::endl;
#     pragma omp task depend(out:deps[1])
      {
        std::cout<<"dep-out-1 task started on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
        std::cout<<"dep-out-1 task end"<<std::endl;
      }

#     pragma omp task depend(out:deps[2])
      {
        std::cout<<"dep-out-2 task started on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(2000) );
        std::cout<<"dep-out-2 task end"<<std::endl;
      }

#     pragma omp task if(0) depend(in:deps[1])
      {
        std::cout<<"dep-in-1 task executes on thread #"<<omp_get_thread_num()<<std::endl;
      // inner task dependency is not dependent on dep-out-2 because tasks are not siblings, even though they use if(0)
#       pragma omp task if(0) depend(in:deps[2])
        std::cout<<"dep-in-2 task executes on thread #"<<omp_get_thread_num()<<std::endl;
      }
      std::cout << "thread #"<<omp_get_thread_num()<<" finished task spawn"<<std::endl;
    }
#   pragma omp taskwait
  }
}


void taskgroup_termination()
{
# pragma omp parallel
  {

#   pragma omp single
    {
#     pragma omp taskgroup
      {
#       pragma omp critical(dbg_mesg)
        std::cout << "thread #"<<omp_get_thread_num()<<" generates tasks"<<std::endl;
        
#       pragma omp task
        {
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-1 started on thread #"<<omp_get_thread_num()<<std::endl;

#         pragma omp task
          {
#           pragma omp critical(dbg_mesg)
            std::cout<<"task-1-1 started on thread #"<<omp_get_thread_num()<<std::endl;
            std::this_thread::sleep_for( std::chrono::milliseconds(2500) );
#           pragma omp critical(dbg_mesg)
            std::cout<<"task-1-1 end"<<std::endl;
          }

          std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-1 end"<<std::endl;
        }

#       pragma omp task
        {
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-2 started on thread #"<<omp_get_thread_num()<<std::endl;
          std::this_thread::sleep_for( std::chrono::milliseconds(1500) );
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-2 end"<<std::endl;
        }

#       pragma omp task
        {
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-3 started on thread #"<<omp_get_thread_num()<<std::endl;
          std::this_thread::sleep_for( std::chrono::milliseconds(2000) );
#         pragma omp critical(dbg_mesg)
          std::cout<<"task-3 end"<<std::endl;
        }

      } // --- end of task group ---
      
#     pragma omp critical(dbg_mesg)
      std::cout << "thread #"<<omp_get_thread_num()<<" finished task spawn"<<std::endl;
    } // --- end of single ---


#   pragma omp single // without a task scheduling point afetr taskgroup, this will (potentially) execute before tasks in taskgroup completed
    {
#     pragma omp critical(dbg_mesg)
      std::cout << "thread #"<<omp_get_thread_num()<<" after task group"<<std::endl;
    }
    
  }
}



void outside_taskgroup_dependence()
{
  int task_A = 0;
  int task_B = 0;
  int task_C = 0;
  int task_D = 0;
  int task_E = 0;
  int task_F = 0;

# pragma omp parallel
  {
#   pragma omp single
    {

#     pragma omp task depend(out:task_A)
      {
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task A start, on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(500) );
        task_A = 1;
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task A end, on thread #"<<omp_get_thread_num()<<std::endl;
      }

#     pragma omp task depend(in:task_A) depend(out:task_B)
      {
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task B start, on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(500) );
        task_B = 1;
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task B end, on thread #"<<omp_get_thread_num()<<std::endl;
      }

#     pragma omp task 
      {
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task Z1 start, on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(2500) );
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task Z1 end, on thread #"<<omp_get_thread_num()<<std::endl;
      }

#     pragma omp taskgroup
      {

#       pragma omp task depend(in:task_B) depend(out:task_C)
        {
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task C start, on thread #"<<omp_get_thread_num()<<std::endl;
          task_C = 1;
          std::this_thread::sleep_for( std::chrono::milliseconds(400) );
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task C end, on thread #"<<omp_get_thread_num()<<std::endl;
        }
        
#       pragma omp task depend(out:task_D)
        {
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task D start"<<", on thread #"<<omp_get_thread_num()<<std::endl;
          std::this_thread::sleep_for( std::chrono::milliseconds(300) );
          task_D = 1;
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task D end"<<", on thread #"<<omp_get_thread_num()<<std::endl;
        }

#       pragma omp task depend(in:task_D)
        {
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task E start"<<", on thread #"<<omp_get_thread_num()<<std::endl;
          task_E= 1;
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task E end"<<", on thread #"<<omp_get_thread_num()<<", task_E="<<task_E<< std::endl;
        }

#       pragma omp task depend(in:task_B,task_C)
        {
          task_F = 1;
          _Pragma("omp critical(dbg_mesg)") std::cout <<"group task F"<<", on thread #"<<omp_get_thread_num()<<", task_F="<<task_F<<std::endl;
        }

      } // end of taskgroup

#     pragma omp task 
      {
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task Z2 start, on thread #"<<omp_get_thread_num()<<std::endl;
        std::this_thread::sleep_for( std::chrono::milliseconds(2000) );
        _Pragma("omp critical(dbg_mesg)") std::cout <<"task Z2 end, on thread #"<<omp_get_thread_num()<<std::endl;
      }

#     pragma omp critical(dbg_mesg)
      std::cout <<"end of task group"<<std::endl;
    }
    
#   pragma omp taskwait

#   pragma omp single
    {
#     pragma omp critical(dbg_mesg)
      std::cout <<"end of lonely tasks"<<std::endl;
    }

  }

}

void taskset_termination_detection()
{
  using std::cout;
  using std::endl;
  
  const int nb_tasks = 100;
  std::atomic<int64_t> taskgroup_counter = 0;

# pragma omp parallel
  {
#   pragma omp single
    {
      
#     pragma omp task default(none) shared(taskgroup_counter,cout)
      {
        int c = taskgroup_counter.load( std::memory_order_relaxed );
        do
        {
          _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" sees "<<taskgroup_counter<<"/"<<nb_tasks<<" completed" <<endl;
          c = taskgroup_counter.load( std::memory_order_relaxed );
        }
        while( c < nb_tasks );
      }

      for(int i=0;i<nb_tasks;i++)
      {
#       pragma omp task default(none) firstprivate(i) shared(taskgroup_counter,cout)
        {
          _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" execute task "<<i <<endl;
          taskgroup_counter.fetch_add( 1 , std::memory_order_relaxed );
        }
      }

    }
  }
}

void task_detach_omp_fulfill()
{
  using std::cout;
  using namespace onika;

  // onika::omp::OpenMPToolInterace::enable_internal_dbg_message();
  // onika::omp::OpenMPToolInterace::enable();

  static constexpr int N = 10;
  volatile int x = 3;
  int k = x;
  size_t dep[N];
  omp_event_handle_t evt[N];

  std::cout<<"sizeof(omp_event_handle_t) = "<<sizeof(omp_event_handle_t)<<std::endl;

# pragma omp parallel
  {
#   pragma omp single
    {
      for(int i=0;i<N;i++)
      {
        omp_event_handle_t tmp_evt;
        static_assert( sizeof(omp_event_handle_t) == sizeof(void*) );
        void** evt_ptr = reinterpret_cast<void**>(&tmp_evt);
        * evt_ptr = nullptr;
        // Note: this fails if we use clause if(0), so it's not possible to save a task creation when empty task is used
        OMP_TASK_DETACH( default(none) firstprivate(i) , depend(out:dep[i]) untied, tmp_evt )
        {
          ONIKA_DBG_MESG_LOCK { stdout_stream() <<"thread #"<<omp_get_thread_num()<<" start detached task "<<i<<std::endl<<std::flush; }
          std::this_thread::sleep_for( std::chrono::milliseconds(500) );
          ONIKA_DBG_MESG_LOCK { stdout_stream() <<"thread #"<<omp_get_thread_num()<<" end detached task "<<i<<" (wait for fulfill)"<< std::endl<<std::flush; }
        }
        ONIKA_DBG_MESG_LOCK { stdout_stream()<<"detach "<<i<<" evt = "<< (*evt_ptr) << " , dep = "<< (void*)(&dep[i]) <<std::endl<<std::flush; }
        evt[i] = std::move(tmp_evt);
      }

#     pragma omp task default(none) firstprivate(k) shared(evt) untied
      {
        std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
        for(int i=0;i<N;i++)
        {
          std::this_thread::sleep_for( std::chrono::milliseconds(500) );
          int j=(k+i)%N;
          ONIKA_DBG_MESG_LOCK { stdout_stream() <<"thread #"<<omp_get_thread_num()<<" fulfill evt "<<j<<std::endl<<std::flush; }
          omp_fulfill_event( evt[j] );
        }
      }

      for(int i=0;i<N;i++)
      {
#       pragma omp task default(none) firstprivate(i) depend(in:dep[i]) untied
        {
          ONIKA_DBG_MESG_LOCK { stdout_stream() <<"thread #"<<omp_get_thread_num()<<" run dependent task "<<i<<std::endl<<std::flush; }
        }
      }
    }
  }

  // onika::omp::OpenMPToolInterace::disable();
}


void taskgroup_in_task()
{
  int group_A=0;

  int task_A1=0;
  int task_B1=0;
  int task_C1=0;

  using std::cout;
  using std::endl;
  
# pragma omp parallel
  {
  
#   pragma omp single
    {    
      if(group_A){} // silently mark as used

      // group A
#     pragma omp task depend(out:group_A)
      {
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start group A" <<endl;
#       pragma omp taskgroup
        {
#         pragma omp task depend(out:task_A1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start task A1" <<endl;
            std::this_thread::sleep_for( std::chrono::milliseconds(500) );
            task_A1=1;
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end task A1" <<endl;
          }

#         pragma omp task depend(in:task_A1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task A2" <<endl;
          }

#         pragma omp task
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task A3" <<endl;
          }

        }
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end group A" <<endl;        
      }

      // group B, depend on gorup A
#     pragma omp task depend(in:group_A)
      {
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start group B" <<endl;
#       pragma omp taskgroup
        {
#         pragma omp task depend(out:task_B1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start task B1" <<endl;
            std::this_thread::sleep_for( std::chrono::milliseconds(500) );
            task_B1=1;
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end task B1" <<endl;
          }

#         pragma omp task depend(in:task_B1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task B2" <<endl;
          }

#         pragma omp task
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task B3" <<endl;
          }

        }
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end group B" <<endl;        
      }

      // group C, no depend
#     pragma omp task
      {
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start group C" <<endl;
#       pragma omp taskgroup
        {
#         pragma omp task depend(out:task_C1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" start task C1" <<endl;
            std::this_thread::sleep_for( std::chrono::milliseconds(500) );
            task_C1=1;
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end task C1" <<endl;
          }

#         pragma omp task depend(in:task_C1)
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task C2" <<endl;
          }

#         pragma omp task
          {
            _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" task C3" <<endl;
          }

        }
        _Pragma("omp critical(dbg_mesg)") cout <<"thread #"<<omp_get_thread_num()<<" end group C" <<endl;        
      }

    }
  }
}


void dynamic_array_firstprivate()
{
  using std::cout;
  //const unsigned int array_sz = 10;

# pragma omp parallel
  {
#   pragma omp single
    {
      // double array[array_sz]; // do not work, seen as a pointer, not copied correctly anyway
      double array[10];
      for(int i=0;i<10;i++) array[i] = i;
#     pragma omp task default(none) shared(cout) firstprivate(array)
      {
        std::this_thread::sleep_for( std::chrono::milliseconds(500) );
        for(int i=0;i<10;i++) cout<<"task value "<<i<<" = "<< array[i]<<std::endl;
      }
      for(int i=0;i<10;i++) array[i] = i+1.01;
      for(int i=0;i<10;i++) std::cout<<"outside value "<<i<<" = "<< array[i]<<std::endl;
    }
  }
}


void tasks_and_locks()
{
  using std::cout;

  auto task_friendly_lock = [](omp_lock_t* lock)
    {
      while( ! omp_test_lock(lock) )
      {
#       pragma omp taskyield
      }
    };

  omp_lock_t lock;
# pragma omp parallel
  {
#   pragma omp single nowait
    {
      omp_init_lock(&lock);
      omp_set_lock(&lock);
#     pragma omp task untied
      {
        //omp_set_lock(&lock);
        task_friendly_lock(&lock);
        cout << "T0 has acquired lock" << std::endl << std::flush;
      }
#     pragma omp task untied
      {
        cout << "T1 releases lock" << std::endl << std::flush;
        omp_unset_lock(&lock);
      }
    }
  }
}


void omp_task_buffer_size(const int MAX_NUMBER = 1000)
{
  int result[MAX_NUMBER];
  for(int i=0;i<MAX_NUMBER;i++) result[i]=0;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  const int nthreads = omp_get_max_threads();
  std::mt19937 gen[nthreads];
  for(int i=0;i<nthreads;i++) gen[i].seed(rd()); //Standard mersenne_twister_engine seeded with rd()
  
# pragma omp parallel
  {
#   pragma omp single
    {
      if(result[0]){}
#     pragma omp task default(none) shared(gen,result) firstprivate(MAX_NUMBER) // generation task
      {
        onika_ompt_begin_task("spawner");
        std::this_thread::sleep_for( std::chrono::milliseconds(100) );
        for(int i=0;i<MAX_NUMBER;i++)
        {
#         pragma omp task default(none) shared(gen,result) firstprivate(i,MAX_NUMBER)
          {
            onika_ompt_begin_task("compute");
            std::uniform_int_distribution<> distrib(0,1000000);
            int x = distrib( gen[omp_get_thread_num()] );
            int p=0;
            while(x>1) { ++p; x = ((x%2)==0) ? (x/2) : (3*x+1); }
            result[i] = p;
            onika_ompt_end_current_task();
          }
        }
        onika_ompt_end_current_task();
      }
    }
  }
}


#define RUN_TEST(n,c) do{ if(test==n||test==-1){std::cout<< "--------- test "<<#n<<" : "<< #c << " -------" << std::endl; c; } }while(0)

int main(int argc,char*argv[])
{
  bool enable_ompt = false;
  int test=-1;
  
  int eargs[64];
  int nea=0;
  
  for(int i=1;i<argc;i++)
  {
    if( std::string(argv[i])=="-ompt" ) enable_ompt=true;
    else if(test==-1) test = std::atoi(argv[i]);
    else eargs[nea++] = std::atoi(argv[i]);
  }
  std::cout<<"OpenMP test #"<<test<<", ompt="<<std::boolalpha<<enable_ompt<<", extra args";
  for(int i=0;i<nea;i++) std::cout<<" "<<eargs[i];
  std::cout <<std::endl;

  onika_ompt_declare_task_context(my_app_ctx);
  void* app_ctx_value = (void*)"omp_test"; if(app_ctx_value==nullptr){}
  if(enable_ompt)
  {
    //onika::omp::OpenMPToolInterace::enable_internal_dbg_message(); // comment this out to avoid tool verbosity
    onika::omp::OpenMPToolInterace::set_app_ctx_printer( my_app_context_printer );
    onika::omp::OpenMPToolInterace::set_task_start_callback( my_task_start_callback );
    onika::omp::OpenMPToolInterace::set_task_stop_callback( my_task_stop_callback );  
    onika::omp::OpenMPToolInterace::enable();
    onika_ompt_begin_task_context2(my_app_ctx,app_ctx_value);
  }

  /*
  the following example shows that for a dependency to be "connected",
  the time relation between task creation is not sufficient.
  */
  RUN_TEST( 1 , task_dependency(1) );
  RUN_TEST( 2 , task_dependency(0) );
  RUN_TEST( 3 , undeferred_task_dependency() );
  RUN_TEST( 4 , undeferred_task_inner_sibling_dependency() );
  RUN_TEST( 5 , taskgroup_termination() );
  RUN_TEST( 6 , outside_taskgroup_dependence() );
  RUN_TEST( 7 , taskset_termination_detection() );
  RUN_TEST( 8 , task_detach_omp_fulfill() );
  RUN_TEST( 9 , taskgroup_in_task() );
  RUN_TEST(10 , dynamic_array_firstprivate() );
  RUN_TEST(11 , tasks_and_locks() );
  RUN_TEST(12 , omp_task_buffer_size(nea>=1?eargs[0]:1000) );
  
  
  if(enable_ompt)
  {
    onika_ompt_end_task_context2(my_app_ctx,app_ctx_value);
    onika::omp::OpenMPToolInterace::disable();
  }

  return 0;
}

