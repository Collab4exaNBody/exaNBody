#pragma once

namespace onika
{
  template<class T, long ... Dims> struct nd_array;
  template<class T> struct nd_array<T> { T data; };
  template<class T, long I> struct nd_array<T,I> { T data[I]; };
  template<class T, long I, long J> struct nd_array<T,I,J> { T data[J][I]; };
  template<class T, long I, long J, long K> struct nd_array<T,I,J,K> { T data[K][J][I]; };
  template<class T, long I, long J, long K, long L> struct nd_array<T,I,J,K,L> { T data[L][K][J][I]; };
}

