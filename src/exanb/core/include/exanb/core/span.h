#pragma once

namespace exanb
{
  template<class T>
  struct span
  {
    inline T* begin() const { return m_begin; }
    inline T* end() const { return m_begin+m_size; }
    T* m_begin = nullptr;
    size_t m_size = 0;
  };
  
  template<class T> inline span<T> make_span(T* p, size_t n) { return span<T>{p,n}; }
  template<class T,class A> inline span<T> make_span(std::vector<T,A>& v, size_t n) { return span<T>{v.data(),n}; }

  template<class T> inline span<const T> make_const_span(const T* p, size_t n) { return span<const T>{p,n}; }
  template<class T,class A> inline span<const T> make_const_span(const std::vector<T,A>& v) { return span<const T>{v.data(),v.size()}; }
}

