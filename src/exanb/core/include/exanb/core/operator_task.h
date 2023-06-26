#pragma once 

#include <exanb/core/operator_slot_reorder.h>
#include <exanb/core/cpp_utils.h>
#include <exanb/field_sets.h>

// uncomment the following 4 lines to print debug messages
/*
#include <onika/task/tag_utils.h>
#include <onika/stream_utils.h>
#define __ONIKA_TASK_SCHED_DBG ONIKA_STDOUT_OSTREAM<<"operator_task : schedule : "<< onika::task::tag_filter_out_path(tag)<<" "; onika::task::print_depend_indices(ONIKA_STDOUT_OSTREAM,plist_t{}) << std::endl; 
#define __ONIKA_TASK_BEGIN_DBG ONIKA_STDOUT_OSTREAM<<"operator_task : start    : "<< onika::task::tag_filter_out_path(tag) << std::endl;
#define __ONIKA_TASK_END_DBG   ONIKA_STDOUT_OSTREAM<<"operator_task : end      : "<< onika::task::tag_filter_out_path(tag) << std::endl;
*/

#include <onika/omp/ompt_interface.h>
#include <onika/task/static_task_scheduler.h>
#include <onika/task/parallel_execution.h>
#include <onika/dac/soatl.h>

#define onika_operator_task(...) \
::onika::task::static_task_scheduler( __FILE__ ":" USTAMP_STR(__LINE__) , ::exanb::forward_slot_data_pointers(__VA_ARGS__) , ::exanb::slot_args_reorder(__VA_ARGS__) ) \
<< [=] ( EXPAND_WITH_PREFIX(auto&,##__VA_ARGS__) )

namespace exanb
{

  // generate a local grid local read-only access stencil from a set of fields
  template<class FieldSetT> struct LocalStencilFromFieldSet;
  template<class... _Fields> struct LocalStencilFromFieldSet< FieldSet<_Fields...> >
  {
    using type = ::onika::dac::local_ro_stencil_t< ::onika::dac::field_array_size_t, _Fields... >;
  };
  template<class T> using local_stencil_from_field_set_t = typename LocalStencilFromFieldSet<T>::type;    

  // generate a local grid local read-write access stencil from a set of fields
  template<class FieldSetT> struct LocalRWStencilFromFieldSet;
  template<class... _Fields> struct LocalRWStencilFromFieldSet< FieldSet<_Fields...> >
  {
    using type = ::onika::dac::local_rw_stencil_t< ::onika::dac::field_array_size_t, _Fields... >;
  };
  template<class T> using local_rw_stencil_from_field_set_t = typename LocalRWStencilFromFieldSet<T>::type;    

  // generate a local grid local read-write access stencil from a set of fields
  template<class FieldSetT> struct DataSlicesFromFieldSet;
  template<class... _Fields> struct DataSlicesFromFieldSet< FieldSet<_Fields...> >
  {
    using type = ::onika::dac::DataSlices<  _Fields ... >;
  };
  template<class T> using slices_from_field_set_t = typename DataSlicesFromFieldSet<T>::type;    
  
}

