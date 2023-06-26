#pragma once

#include <functional>
#include <string_view>
#include <cstdlib>

namespace onika
{
  template<class T> struct ignore_value_t { using type=T; };
  template<class T> static inline constexpr ignore_value_t<T> ignore_value(const T&) { return {}; }
  template<class T> struct IsValueIgnored : public std::false_type {};
  template<class T> struct IsValueIgnored< ignore_value_t<T> > : public std::true_type {};
  template<class T> static inline constexpr bool is_value_ignored_v = IsValueIgnored<T>::value;
}

namespace std
{
  template<class T> struct hash< onika::ignore_value_t<T> >
  {
    inline size_t operator () ( const onika::ignore_value_t<T> & ) const { return 0; }
  };
}

namespace onika
{
  template<class... Args>
  static inline size_t multi_hash(const Args& ... args)
  {
    size_t TH[ sizeof...(args) * 2 ];
    ssize_t i = 0;
    ( ... , ( TH[i++] = typeid(Args).hash_code() , (!is_value_ignored_v<Args>) ? TH[i++] = std::hash<Args>{}(args) : TH[i] ) );
    assert( i>=ssize_t(sizeof...(args)) && i<=ssize_t(sizeof...(args)*2) );
    return std::hash<std::string_view>{}( std::string_view( (const char*) TH , i * sizeof(size_t) ) );
  }
}
