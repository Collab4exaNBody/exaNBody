#pragma once

#include <onika/dac/stencil.h>
#include <onika/dac/box_span.h>

#include <onika/dag/dag.h>
#include <onika/dag/dag_algorithm.h>
#include <onika/dag/dag_filter.h>
#include <onika/dag/dag_reorder.h>
#include <onika/dag/dag_execution.h>
#include <onika/hash_utils.h>
#include <onika/task/parallel_task_cost.h>
#include <onika/task/ptask_proxy_proto.h>
#include <onika/omp/task_detach.h>
#include <onika/omp/dynamic_depend_dispatch.h>
#include <onika/lambda_tools.h>

#include <onika/dac/box_span_stream.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace onika
{

  namespace task
  {
    template<class SpanT, TaskTypeEnum _TaskType, class AccTuple, class KernelFunc, class CostFunc, bool _CudaEnabled> struct ParallelTaskExecutorImpl;

    struct ParallelTaskFunctorMem
    {
      void* kernel_func = nullptr;
      size_t kernel_size = 0;
      size_t kernel_hash = 0;
      void* cost_func = nullptr;
      size_t cost_size = 0;
      size_t cost_hash = 0;
    };

    // bridge between template specialized and specialization oblivious code parts
    struct ParallelTaskExecutor
    {
      static inline constexpr size_t MAX_ACCESSED_DATA_COUNT = 16;

      // abstract interface, implented in ParallelTaskExecutorImpl specialization
      virtual void invalidate_graph() =0;
      virtual const dag::AbstractWorkShareDAG& dep_graph() const =0;
      virtual bool build_graph(ParallelTask* pt) =0;

      //virtual oarray_t<size_t,1> task_coord_1( ParallelTask* pt, size_t i) const =0;
      //virtual oarray_t<size_t,2> task_coord_2( ParallelTask* pt, size_t i) const =0;
      //virtual oarray_t<size_t,3> task_coord_3( ParallelTask* pt, size_t i) const =0;

      virtual void execute( ParallelTask* pt ) const =0;
      virtual void execute( ParallelTask* pt , size_t i ) const =0;
      virtual void execute( ParallelTask* pt , oarray_t<size_t,1> c ) const =0;
      virtual void execute( ParallelTask* pt , oarray_t<size_t,2> c ) const =0;
      virtual void execute( ParallelTask* pt , oarray_t<size_t,3> c ) const =0;

      virtual void spawn_omp_all_tasks( ParallelTask* pt ) =0;
      //virtual void spawn_omp_task( ParallelTask* pt , size_t i ) const =0;
      
      virtual size_t dag_skipped_elements() const =0;    
      virtual size_t dag_span_elements() const =0;
      
      virtual void all_tasks_completed_notify() =0;
      
      virtual const char* tag() const =0;
      // --------------------

      template<class SpanT, class AccTuple, class NdISeq, class ODIseq>
      inline ParallelTaskExecutor(size_t czh, size_t dh, size_t sh, const SpanT& sp, const AccTuple& accessors, NdISeq , ODIseq )
        : m_abstract_span(sp)
        , m_abstract_stencil(dac::dac_set_nd_subset(accessors))
        , m_codezone_hash(czh)
        , m_dac_hash(dh)
        , m_span_hash(sh)
      {
        copy_accessor_pointers( accessors , NdISeq{} );
        copy_scalar_pointers( accessors , ODIseq{} );  
      }

      /*inline ~ParallelTaskExecutor()
      {
        std::cout<<"~ParallelTaskExecutor @"<<(void*)this<<"\n";
      }*/

      inline const dac::abstract_box_span_t& span() const { return m_abstract_span; }
      inline const dac::AbstractStencil& stencil() const { return m_abstract_stencil; }
      inline std::vector<omp_event_handle_t>& omp_completion_events() { return m_omp_completion_events; }

      /*template<size_t Nd>
      inline oarray_t<size_t,Nd> task_coord( ParallelTask* pt, size_t i , ConstDim<Nd> ) const
      {
        static_assert( Nd<=3 , "Only dimensions up to 3 are supported" );
        if constexpr ( Nd == 0 ) return {};
        if constexpr ( Nd == 1 ) return task_coord_1(pt,i);
        if constexpr ( Nd == 2 ) return task_coord_2(pt,i);
        if constexpr ( Nd == 3 ) return task_coord_3(pt,i);
        std::abort(); return {};
      }*/

      static void clear_ptexecutor_cache();

      template<class _Span, TaskTypeEnum _TaskType, class AccTuple, class _TaskFunc, class _CostFunc, bool _CudaEnabled>
      static inline std::shared_ptr<ParallelTaskExecutor> ptexecutor_adapter( PTaskProxy<_Span,_TaskType,AccTuple,_TaskFunc,_CostFunc,_CudaEnabled> && proxy )
      {
        using dac_set_nd_indices = typename dac::DacSetSplitter<AccTuple>::dac_set_nd_indices;
        using dac_set_0d_indices = typename dac::DacSetSplitter<AccTuple>::dac_set_0d_indices;
        using PTaskExecImplT = ParallelTaskExecutorImpl<_Span,_TaskType,AccTuple,_TaskFunc,_CostFunc,_CudaEnabled>;

        auto [ codezone_hash , dac_hash , span_hash ] = ptexecutor_hash(proxy.m_tag,proxy.m_span,proxy.m_accs,proxy.m_func,proxy.m_cost,dac_set_nd_indices{});

        std::shared_ptr<ParallelTaskExecutor> sptr;

        std::scoped_lock lock(s_ptexecutor_cache_mutex); // critical section
        {
          // std::cout << format_box_span(proxy.m_span) << " -> "<<span_hash<<"\n";
          // std::cout << "look for ptexecutor : CH="<<codezone_hash<<", DH="<<dac_hash<<", SH="<<span_hash<<", tag="<<proxy.m_tag<<", ptq="<<(void*)proxy.m_queue<< "\n";
          auto it = s_ptexecutor_cache.find(codezone_hash);
          if( it == s_ptexecutor_cache.end() )
          {
  	        //std::cout << "create ptexecutor span="<<format_box_span(proxy.m_span)<<", tag="<<proxy.m_tag<<"\n";
            sptr = std::make_shared<PTaskExecImplT>( codezone_hash, dac_hash , span_hash , std::move(proxy) );
            s_ptexecutor_cache.insert( { codezone_hash , sptr } );
          }
          else
          {
  	        //std::cout << "update ptexecutor span="<<format_box_span(proxy.m_span)<<", tag="<<proxy.m_tag<<", ptq="<<(void*)proxy.m_queue<<"\n";
            sptr = it->second;
            sptr->update_dac( dac_hash , std::move(proxy.m_accs) , dac_set_nd_indices{} , dac_set_0d_indices{} );
            sptr->update_span( span_hash , std::move(proxy.m_span) );
            sptr->update_functors( std::move(proxy.m_func) , std::move(proxy.m_cost) );
            sptr->update_ptask( proxy.m_ptask );
            proxy.reset(); // same effect as move contructed proxy from this one
          }
        }
        
        //size_t refcount = sptr.use_count();
        //std::cout << "ptexecutor refcount = " << refcount << std::endl;
        assert( sptr.use_count() == 2 ); // the one in map and the one in sptr, must not be used elsewhere
        assert( sptr->codezone_hash() == codezone_hash );

        return sptr;
      }

      inline size_t data_stencil_bits(size_t i) const
      {
        size_t s = m_stencil_shift[i];
        size_t e = m_stencil_bits;
        if( i < (m_data_count-1) ) e = m_stencil_shift[i+1];
        return e-s;
      }

      template<class AccTuple, class NdISeq, class ODIseq>
      inline void update_dac( size_t dac_hash , AccTuple && accessors, NdISeq , ODIseq )
      {
        if( dac_hash != m_dac_hash )
        {
          //std::cout << "PTDAP: update dac" << std::endl;
          m_dac_hash = dac_hash;
          copy_accessor_pointers( accessors , NdISeq{} );
          copy_scalar_pointers( accessors , ODIseq{} );  
        }
      }

      virtual void update_span_int() =0;
      virtual void update_ptask( ParallelTask* pt ) = 0;
      
      template<class SpanT>
      inline void update_span( size_t span_hash , SpanT && sp )
      {
        if( span_hash != m_span_hash )
        {
          //std::cout << "PTDAP: update span" << std::endl;
          m_abstract_span = dac::abstract_box_span_t(sp);
          this->update_span_int();
          invalidate_graph();
        }
#       ifndef NDEBUG
        else
        {
          auto asp = dac::abstract_box_span_t(sp);
          assert( m_abstract_span == asp );
        }
#       endif
      }

    protected:
      virtual ParallelTaskFunctorMem get_functors_mem() =0;     

      template<class AccTuple, size_t... Is>
      inline void copy_accessor_pointers( const AccTuple& accessors, std::integer_sequence<size_t,Is...> )
      {
        static_assert( sizeof...(Is) < MAX_ACCESSED_DATA_COUNT );
        size_t i = 0;
        ( ... , ( m_data_ptr[i++] = accessors.get(tuple_index<Is>).pointer() ) );
        m_data_count = i;
        assert( m_data_count == sizeof...(Is) );
        i = 0;
        m_stencil_bits = 0;
        ( ... , ( m_stencil_shift[i++] = m_stencil_bits , m_stencil_bits += accessors.get(tuple_index<Is>).nb_slices ) );
        assert( i == sizeof...(Is) );
      }

      template<class AccTuple, size_t... Is>
      inline void copy_scalar_pointers( const AccTuple& accessors, std::integer_sequence<size_t,Is...> )
      {
        static_assert( sizeof...(Is) < MAX_ACCESSED_DATA_COUNT );
        m_scalar_count = 0;
        if constexpr ( sizeof...(Is) != 0 )
        {
          ( ... , ( m_scalar_ptr[m_scalar_count++] = accessors.get(tuple_index<Is>).pointer() ) );
        }
      }
    
      virtual const char* get_tag() const =0;
      inline std::size_t codezone_hash() const { return m_codezone_hash; }
      inline std::size_t dac_hash() const { return m_dac_hash; }
      inline std::size_t span_hash() const { return m_span_hash; }
      
      // protected members
      //std::vector<uint16_t> m_dep_depth; // maximum dependence depth for ith task
      size_t m_max_dep_depth = 0;
      
    private:
      template<class _Span, class AccTuple, class KernelFunc, class CostFunc, size_t... Is>
      static inline std::array<size_t,3> ptexecutor_hash(const char* tag, const _Span& sp, const AccTuple& accs, const KernelFunc & kf, const CostFunc & cf, std::integer_sequence<size_t,Is...>)
      {
        return { multi_hash(tag,ignore_value(sp),ignore_value(kf),ignore_value(cf),ignore_value(accs.get(tuple_index<Is>))...) , multi_hash(accs.get(tuple_index<Is>)...) , multi_hash(sp) };
      }

      template<class KernelFunc, class CostFunc>
      inline void update_functors( KernelFunc && kfunc , CostFunc && costfunc )
      {
        auto fm = get_functors_mem();
        assert( fm.kernel_size == sizeof(KernelFunc) && fm.kernel_hash == typeid(KernelFunc).hash_code() );
        assert( fm.cost_size == sizeof(CostFunc) && fm.cost_hash == typeid(CostFunc).hash_code() );
        
        auto kp = reinterpret_cast< KernelFunc * >( fm.kernel_func );
        auto cp = reinterpret_cast< CostFunc * >( fm.cost_func );

        kp -> ~KernelFunc();
        kp = new(fm.kernel_func) KernelFunc( std::move(kfunc) );

        cp -> ~CostFunc();        
        cp = new(fm.cost_func) CostFunc( std::move(costfunc) );        

        assert( kp == fm.kernel_func );
        assert( cp == fm.cost_func );
      }

      // instance members
      dac::abstract_box_span_t m_abstract_span;
      dac::AbstractStencil m_abstract_stencil;
      std::vector<omp_event_handle_t> m_omp_completion_events;

      const void* m_data_ptr[MAX_ACCESSED_DATA_COUNT];
      const void* m_scalar_ptr[MAX_ACCESSED_DATA_COUNT];
      size_t m_stencil_shift[MAX_ACCESSED_DATA_COUNT];      
      size_t m_stencil_bits = 0; // may be greater than m_stencil.m_nbits
      size_t m_scalar_count = 0;
      size_t m_data_count = 0;

      size_t m_codezone_hash = 0; // tag + all template types, without values
      size_t m_dac_hash = 0; // stencil's value + accessors' values
      size_t m_span_hash = 0; // stencil's value + span's value
      
      // class members
      static std::mutex s_ptexecutor_cache_mutex;
      static std::unordered_map< std::size_t , std::shared_ptr<ParallelTaskExecutor> > s_ptexecutor_cache;
    };


  }
}


