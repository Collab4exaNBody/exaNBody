#include <exanb/field_sets.h>
#include <exanb/core/type_utils.h>
#include "test_field_set.h"

#include <onika/variadic_template_utils.h>

#include <iostream>
#include <tuple>

template<typename fs> struct GridFromFieldSet {};
template<typename... fs> struct GridFromFieldSet< exanb::FieldSet<fs...> >
{
  using GridType = DummyGrid<fs...>;
};

template<typename fs> struct MakeGridTuple {};
template<typename... fs> struct MakeGridTuple< exanb::FieldSets<fs...> >
{
  using GridTuple = std::tuple< typename GridFromFieldSet<fs>::GridType ... >;
};



template< template<typename> typename C , typename field_set > struct InstantiateWithFieldSet
{
  using type = typename C< field_set >::GridType;
};

template<typename T> std::string pretty_type()
{
  return exanb::strip_type_spaces( exanb::remove_exanb_namespaces( exanb::type_as_string<T>() ) );
}

template<typename fs,typename subtract_fs> struct RemoveFromFieldsTester {};
template<typename subtract_fs, typename... fs> struct RemoveFromFieldsTester< exanb::FieldSets<fs...> , subtract_fs >
{
  inline RemoveFromFieldsTester()
  {
    TEMPLATE_LIST_BEGIN
      std::cout << pretty_type<fs>() << " - " << pretty_type<subtract_fs>() << " = " << pretty_type< exanb::RemoveFields<fs,subtract_fs> >() << std::endl
    TEMPLATE_LIST_END
  }
};

template<typename fs> struct AddDefaultFieldsTester {};
template<typename... fs> struct AddDefaultFieldsTester< exanb::FieldSets<fs...> >
{
  inline AddDefaultFieldsTester()
  {
    TEMPLATE_LIST_BEGIN
      std::cout << pretty_type<fs>() << " + <defaults> = " << pretty_type< exanb::AddDefaultFields<fs> >() << std::endl
    TEMPLATE_LIST_END
  }
};


int main()
{
  using namespace exanb::field;
  
  using G1 = DummyGrid<_ep,_ax,_ay,_az,_vx,_vy,_vz>;
  using G2 = DummyGrid<_ep,_ax,_ay,_az,_vx,_vy,_vz,_id,_type>;
  using G3 = DummyGrid<_ep,_ax,_ay,_az,_vx,_vy,_vz,_id,_type,_virial,_idmol,_cmol,_charge>;
  
  G1 g1;
  G2 g2;
  G3 g3;
  
  std::cout << G1::dummy_grid_function_a() << std::endl;
  std::cout << G2::dummy_grid_function_a() << std::endl;
  std::cout << G3::dummy_grid_function_a() << std::endl;

  std::cout << g1.dummy_grid_function_b() << std::endl;
  std::cout << g2.dummy_grid_function_b() << std::endl;
  std::cout << g3.dummy_grid_function_b() << std::endl;

  // now define a tuple of G1,G2,G3 using StandardFieldSets
  typename MakeGridTuple< exanb::StandardFieldSets >::GridTuple tp;
  
  std::cout << std::get<0>(tp).dummy_grid_function_b() << std::endl;
  std::cout << std::get<1>(tp).dummy_grid_function_b() << std::endl;
  std::cout << std::get<2>(tp).dummy_grid_function_b() << std::endl;

  using MyFieldSet = exanb::FieldSet<_ep,_ax,_ay,_az,_vx,_vy,_vz>;

  typename InstantiateWithFieldSet< GridFromFieldSet , MyFieldSet >::type g4;
  std::cout << typeid(g4).name() << std::endl;
  
  std::cout << pretty_type<MyFieldSet>() << " contains vx = " << std::boolalpha << exanb::FieldSetContainField<MyFieldSet,_vx>::value << std::endl;
  std::cout << pretty_type<MyFieldSet>() << " contains id = " << std::boolalpha << exanb::FieldSetContainField<MyFieldSet,_id>::value << std::endl;

  // sample requirements
  using R1 = exanb::FieldSet<_rx,_ry,_rz>;
  using R2 = exanb::FieldSet<_vx,_vy,_vz>;
  using R3 = exanb::FieldSet<_rx,_ry,_vz>;
  using R4 = exanb::FieldSet<_vx,_id,_vz>;

  std::cout << pretty_type<MyFieldSet>() << " contains " << pretty_type<R1>() << " = " << std::boolalpha << exanb::FieldSetContainFieldSet<MyFieldSet,R1>::value << std::endl;
  std::cout << pretty_type<MyFieldSet>() << " contains " << pretty_type<R2>() << " = " << std::boolalpha << exanb::FieldSetContainFieldSet<MyFieldSet,R2>::value << std::endl;
  std::cout << pretty_type<MyFieldSet>() << " contains " << pretty_type<R3>() << " = " << std::boolalpha << exanb::FieldSetContainFieldSet<MyFieldSet,R3>::value << std::endl;
  std::cout << pretty_type<MyFieldSet>() << " contains " << pretty_type<R4>() << " = " << std::boolalpha << exanb::FieldSetContainFieldSet<MyFieldSet,R4>::value << std::endl;

  std::cout << "standard field sets = " << pretty_type<exanb::StandardFieldSets>() << std::endl;

  using MyFieldSets = exanb::FieldSets<
      exanb::FieldSet<_ep,_ax,_ay,_az,_vx,_vy,_vz>
    , exanb::FieldSet<_ax,_ay,_az,_vx,_vy,_vz,_idmol>
    , exanb::FieldSet<_ep,_ax,_ay,_az,_vx,_vy,_vz,_id,_type>
    , exanb::FieldSet<_ep,_ax,_ay,_az,_vx,_vy,_vz,_id,_type,_idmol>
    >;

  using R5 = exanb::FieldSet<_rx,_ep,_ry,_rz>;
  using R6 = exanb::FieldSet<_idmol>;
  using R7 = exanb::FieldSet<_type>;
  using R8 = exanb::FieldSet<_ep,_idmol>;
  using R0 = exanb::FieldSet<>;

  std::cout << "test sets => " << pretty_type<MyFieldSets>() << std::endl;

  using FS5 = typename exanb::FilterFieldSets< MyFieldSets , R5 >::type;
  using FS6 = typename exanb::FilterFieldSets< MyFieldSets , R6 >::type;
  using FS7 = typename exanb::FilterFieldSets< MyFieldSets , R7 >::type;
  using FS8 = typename exanb::FilterFieldSets< MyFieldSets , R8 >::type;
  using FS0 = typename exanb::FilterFieldSets< MyFieldSets , R0 >::type;

  std::cout << "filter "<< pretty_type<R5>() << " => " << pretty_type<FS5>() << std::endl;
  std::cout << "filter "<< pretty_type<R6>() << " => " << pretty_type<FS6>() << std::endl;
  std::cout << "filter "<< pretty_type<R7>() << " => " << pretty_type<FS7>() << std::endl;
  std::cout << "filter "<< pretty_type<R8>() << " => " << pretty_type<FS8>() << std::endl;
  std::cout << "filter "<< pretty_type<R0>() << " => same as original = " << std::boolalpha << std::is_same<FS0,MyFieldSets>::value << std::endl;

  RemoveFromFieldsTester< MyFieldSets , R5 >();
  RemoveFromFieldsTester< MyFieldSets , R8 >();

  AddDefaultFieldsTester< exanb::FieldSets<R1,R2,R3,R5,R6> >();

  return 0;
}

