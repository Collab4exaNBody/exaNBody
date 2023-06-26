#include <iostream>

#include <onika/omp/ompt_interface.h>
#include <onika/silent_use.h>
#include <omp.h>


struct MyApplicationContext
{
  const char* my_lib_name = nullptr;
  const char* my_compute_kernel = nullptr;
};

static std::ostream& my_app_context_printer( void* ctx , std::ostream& out )
{
  MyApplicationContext* appctx = reinterpret_cast<MyApplicationContext*>( ctx );
  if(appctx!=nullptr && appctx->my_lib_name!=nullptr) out << appctx->my_lib_name;
  else out << "null";
  out << "/";
  if(appctx!=nullptr && appctx->my_compute_kernel!=nullptr) out << appctx->my_compute_kernel;
  else out << "null";
  return out;
}

static void my_task_start_callback( const onika::omp::OpenMPToolTaskTiming& evt )
{
  ONIKA_DBG_MESG_LOCK
  {
    std::cout << "user task start callback : timepoint="<< evt.timepoint.count() << std::endl;
  }
}

static void my_task_stop_callback( const onika::omp::OpenMPToolTaskTiming& evt )
{
  ONIKA_DBG_MESG_LOCK
  {
    std::cout << "user task stop callback : elapsed="<< evt.elapsed().count() << std::endl;
  }
}


int main()
{
  onika::omp::OpenMPToolInterace::enable_internal_dbg_message(); // comment this out to avoid tool verbosity
  
  onika::omp::OpenMPToolInterace::set_app_ctx_printer( my_app_context_printer );
  onika::omp::OpenMPToolInterace::set_task_start_callback( my_task_start_callback );
  onika::omp::OpenMPToolInterace::set_task_stop_callback( my_task_stop_callback );
  
  onika::omp::OpenMPToolInterace::enable();

  // add application specific information about code portion emmiting tasks
  MyApplicationContext appctx = { "computelib" , "mykernel" }; ONIKA_SILENT_USE(appctx);
  onika_ompt_begin_task_context( &appctx );

# pragma omp parallel
  {
/*
#   pragma omp critical
    {
      std::cout<<"thread "<<omp_get_thread_num()<<" among "<<omp_get_num_threads()<<std::endl;
    }
*/
#   pragma omp for
    for(int i=0;i<32;i++)
    {
      std::cout<<"for: thread #"<<omp_get_thread_num()<<" process i="<<i<<std::endl;
    }
    
//#   pragma omp single
    {
#     pragma omp taskloop
      for(int i=0;i<32;i++)
      {
        std::cout<<"taskloop: thread #"<<omp_get_thread_num()<<" process i="<<i<<std::endl;
      }
    }
    
  }

  onika_ompt_end_task_context( &appctx );

  onika::omp::OpenMPToolInterace::disable();

  return 0;
}

