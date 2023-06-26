#pragma once

#include <utility>
#include <type_traits>

namespace onika
{

  template<class T, unsigned int N , class=void > struct aggregate_members_at_least : public std::false_type {};
  template<class T> struct aggregate_members_at_least< T , 1 , decltype(void(sizeof(T{{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 2 , decltype(void(sizeof(T{{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 3 , decltype(void(sizeof(T{{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 4 , decltype(void(sizeof(T{{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 5 , decltype(void(sizeof(T{{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 6 , decltype(void(sizeof(T{{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 7 , decltype(void(sizeof(T{{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 8 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T , 9 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,10 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,11 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,12 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,13 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,14 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,15 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,16 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,17 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,18 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,19 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T> struct aggregate_members_at_least< T ,20 , decltype(void(sizeof(T{{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}))) > : public std::true_type {};
  template<class T,unsigned int N> static inline constexpr bool aggregate_members_at_least_v = aggregate_members_at_least<T,N>::value;

  template<class T>
  static inline constexpr ssize_t aggregate_member_count()
  {
    if constexpr ( aggregate_members_at_least_v<T,20>) { return -1; }
    else if constexpr ( aggregate_members_at_least_v<T,19>) { return 19; }
    else if constexpr ( aggregate_members_at_least_v<T,18>) { return 18; }
    else if constexpr ( aggregate_members_at_least_v<T,17>) { return 17; }
    else if constexpr ( aggregate_members_at_least_v<T,16>) { return 16; }
    else if constexpr ( aggregate_members_at_least_v<T,15>) { return 15; }
    else if constexpr ( aggregate_members_at_least_v<T,14>) { return 14; }
    else if constexpr ( aggregate_members_at_least_v<T,13>) { return 13; }
    else if constexpr ( aggregate_members_at_least_v<T,12>) { return 12; }
    else if constexpr ( aggregate_members_at_least_v<T,11>) { return 11; }
    else if constexpr ( aggregate_members_at_least_v<T,10>) { return 10; }
    else if constexpr ( aggregate_members_at_least_v<T,9> ) { return 9; }
    else if constexpr ( aggregate_members_at_least_v<T,8> ) { return 8; }
    else if constexpr ( aggregate_members_at_least_v<T,7> ) { return 7; }
    else if constexpr ( aggregate_members_at_least_v<T,6> ) { return 6; }
    else if constexpr ( aggregate_members_at_least_v<T,5> ) { return 5; }
    else if constexpr ( aggregate_members_at_least_v<T,4> ) { return 4; }
    else if constexpr ( aggregate_members_at_least_v<T,3> ) { return 3; }
    else if constexpr ( aggregate_members_at_least_v<T,2> ) { return 2; }
    else if constexpr ( aggregate_members_at_least_v<T,1> ) { return 1; }
    return 0;
  }


  // ====================================================
  // count template *type* parameters in a template class instanciation
  // ====================================================
  template<class T> struct TemplateArgsSizeHelper;
  template< template<class... U> class Tmpl , class... T > struct TemplateArgsSizeHelper< Tmpl<T...> > { static inline constexpr size_t value = sizeof...(T); };
  template<class T> static inline constexpr size_t template_args_count_v = TemplateArgsSizeHelper<T>::value;

}


