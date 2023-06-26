#pragma once

namespace onika
{

  namespace dac
  {
  
    // reduction operation available for scalar (0D) values
    struct reduction_add_t {}; static inline constexpr reduction_add_t reduction_add{};
    struct concurrent_write_t {}; static inline constexpr concurrent_write_t concurrent_write{};

    template<class T , class Op = std::nullptr_t > struct ReductionSelector {};
    template<class T > struct ReductionSelector<T,reduction_add_t>
    {
      static inline constexpr T init_value = T(0);
      static inline void reduce(T& a, const T& b) { _Pragma("omp atomic update") a += b; }
    };
    
    template<class T, class Op>
    struct ReductionWrapper
    {
      using wrapper_type = ReductionWrapper<T,Op>;
      using operator_t = Op;
      T& m_shared;
      T m_tmp = ReductionSelector<T,Op>::init_value;
//      inline ReductionWrapper(T* p) : m_shared(p) { _Pragma("omp critical(dbg_mesg)") std::cout<<"init ReductionWrapper"<<std::endl; }
//      inline ReductionWrapper(const ReductionWrapper & rw) : m_shared(rw.m_shared) , m_tmp(rw.m_tmp) { _Pragma("omp critical(dbg_mesg)") std::cout<<"copy ReductionWrapper"<<std::endl; }
//      inline ReductionWrapper(ReductionWrapper && rw) : m_shared(std::move(rw.m_shared)) , m_tmp(std::move(rw.m_tmp)) { _Pragma("omp critical(dbg_mesg)") std::cout<<"move ReductionWrapper"<<std::endl; }
      inline operator const T&() const { return m_tmp; }
      inline T& operator = (const T& r) { m_tmp=r; return m_tmp; }
      inline T& operator += (const T& r) { m_tmp+=r; return m_tmp; }
      inline ~ReductionWrapper()
      {
//        _Pragma("omp critical(dbg_mesg)") std::cout<<"reduce ReductionWrapper"<<std::endl;
        ReductionSelector<T,operator_t>::reduce(m_shared,m_tmp);
      }
    };

    template<class T>
    struct ReductionWrapper<T,concurrent_write_t>
    {
      using wrapper_type = T & ;
      using operator_t = concurrent_write_t;
    };

    template<class T, class Op> using reduction_wrapper_t = typename ReductionWrapper<T,Op>::wrapper_type;

  }

}


