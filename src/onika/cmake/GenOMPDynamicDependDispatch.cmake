function(GenOMPDynamicDependDispatch outfile maxdeps)

  set(code "${code}#include <type_traits>\n")
  set(code "${code}#include <onika/omp/dynamic_depend_dispatch.h>\n")
  set(code "${code}#include <onika/lambda_tools.h>\n")
  set(code "${code}#include <onika/task/task_pool_item.h>\n")
  set(code "${code}#include <onika/task/parallel_task_executor.h>\n")

#  set(code "${code}#pragma GCC diagnostic push\n")
#  set(code "${code}#pragma GCC diagnostic ignored \"-Wunused-variable\"\n")

  set(code "${code}namespace onika\n")
  set(code "${code}{\n")
  set(code "${code}namespace omp\n")
  set(code "${code}{\n")

  set(code "${code}#define CONCAT_STR(x) _CONCAT_STR(x)\n")
  set(code "${code}#define _CONCAT_STR(x) #x\n\n")
  
  set(code "${code}#ifndef ONIKA_DYNAMIC_DEPEND_OMP_TASK_ATTRIBUTES\n")
  set(code "${code}#define ONIKA_DYNAMIC_DEPEND_OMP_TASK_ATTRIBUTES /**/\n")
  set(code "${code}#endif\n")
  
  set(code "${code}  static const DynamicDependOpenMPDispatch null_dynamic_depend_dispatch{};\n\n")
  
  foreach(ndeps RANGE ${maxdeps})
    math(EXPR ndeps_1 "${ndeps}-1")

    set(code "${code}\n  // specializations for ${ndeps} total dependencies\n")

    foreach(nindeps RANGE ${ndeps})
      math(EXPR noutdeps "${ndeps}-${nindeps}")
      math(EXPR noutdeps_1 "${noutdeps}-1")
      math(EXPR nindeps_1 "${nindeps}-1")

      set(code "${code}\n  // specializations for ${nindeps} in-dependencies\n")
      set(code "${code}  struct DynamicDependOpenMPDispatch_${ndeps}_${nindeps} : public DynamicDependOpenMPDispatch\n")
      set(code "${code}  {\n")
      set(code "${code}    template<class Arg0T, class Arg1T, class Arg2T>\n")
      set(code "${code}    inline void exec(const void ** indeps, void ** outdeps, Arg0T arg0, Arg1T arg1, Arg2T arg2) const \n")
      set(code "${code}    {\n")
      set(code "${code}      using auto_free_tpi = std::conditional_t< std::is_same_v<Arg0T,task::TaskPoolItem*> , std::true_type , std::false_type >;\n")
      set(code "${code}      using one_arg = std::conditional_t< std::is_same_v<Arg2T,std::nullptr_t> , std::true_type , std::false_type >;\n")
      set(code "${code}      using two_args = std::conditional_t< ! std::is_same_v<Arg2T,std::nullptr_t> , std::true_type , std::false_type >;\n")
      
      if(${nindeps} GREATER 0)
        set(code "${code}      const int")
        foreach(i RANGE ${nindeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code} *_id${i}=(const int*)(indeps[${i}])")
        endforeach()
        set(code "${code};\n")
      endif()

      if(${noutdeps} GREATER 0)
        set(code "${code}      int")
        foreach(i RANGE ${noutdeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code} *_od${i}=(int*)(outdeps[${i}])")
        endforeach()
        set(code "${code};\n")
      endif()

      set(code "${code}#     pragma omp task default(none) firstprivate(arg0,arg1,arg2)")

      if(${nindeps} GREATER 0)
        set(code "${code} depend(in:")
        foreach(i RANGE ${nindeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}_id${i}[0]")
        endforeach()
        set(code "${code})")
      endif()

      if(${noutdeps} GREATER 0)
        set(code "${code} depend(out:")
        foreach(i RANGE ${noutdeps_1})
          if(${i} GREATER 0)
            set(code "${code},")
          endif()
          set(code "${code}_od${i}[0]")
        endforeach()
        set(code "${code})")
      endif()

      set(code "${code} ONIKA_DYNAMIC_DEPEND_OMP_TASK_ATTRIBUTES\n")
      set(code "${code}      {\n")
      set(code "${code}        if constexpr ( auto_free_tpi ::value ) { arg0->execute(); arg0->free(); }\n")
      set(code "${code}        if constexpr ( one_arg       ::value ) { arg0->execute(arg1); }\n")
      set(code "${code}        if constexpr ( two_args      ::value ) { arg0->execute(arg1,arg2); }\n")
      set(code "${code}      }\n")
      
      set(code "${code}    }\n")
      set(code "${code}    inline void operator () (const void ** indeps, void ** outdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt ) const override final { exec(indeps,outdeps,ptexecutor,pt,nullptr); }\n")
      set(code "${code}    inline void operator () (const void ** indeps, void ** outdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,1> c ) const override final { exec(indeps,outdeps,ptexecutor,pt,c); }\n")
      set(code "${code}    inline void operator () (const void ** indeps, void ** outdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,2> c ) const override final { exec(indeps,outdeps,ptexecutor,pt,c); }\n")
      set(code "${code}    inline void operator () (const void ** indeps, void ** outdeps, const task::ParallelTaskExecutor * ptexecutor, task::ParallelTask* pt, oarray_t<size_t,3> c ) const override final { exec(indeps,outdeps,ptexecutor,pt,c); }\n")
      set(code "${code}    inline void operator () (const void ** indeps, void ** outdeps, task::TaskPoolItem* tpi ) const override final { exec(indeps,outdeps,tpi,nullptr,nullptr); }\n")
      set(code "${code}  };\n")
      set(code "${code}  static_assert( sizeof(DynamicDependOpenMPDispatch_${ndeps}_${nindeps}) == sizeof(DynamicDependOpenMPDispatch) , \"unexpected type size\");\n")
      set(code "${code}  static const DynamicDependOpenMPDispatch_${ndeps}_${nindeps} dynamic_depend_dispatch_${ndeps}_${nindeps}{};\n")
    endforeach()

  endforeach()

  set(code "${code}\n  const DynamicDependOpenMPDispatch* dynamic_depend_dispatch_table [] = { \n")
  foreach(ndeps RANGE ${maxdeps})
    foreach(nindeps RANGE ${ndeps})
      set(code "${code}\n    &dynamic_depend_dispatch_${ndeps}_${nindeps},\n")
    endforeach()
  endforeach()
    set(code "${code}\n    &null_dynamic_depend_dispatch };\n")

  set(code "${code}\n}\n}\n")
#  set(code "${code}#pragma GCC diagnostic pop\n")

  message(STATUS "generate file ${outfile}")
  file(WRITE ${outfile} "${code}")
endfunction()


