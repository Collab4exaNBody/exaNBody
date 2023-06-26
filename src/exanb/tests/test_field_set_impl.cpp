#include <exanb/field_sets.h>
#include <onika/variadic_template_utils.h>

#include "test_field_set.h"

#include <tuple>
#include <memory>

template<typename... field_ids>
int DummyGrid<field_ids...>::dummy_grid_function_a() { return sizeof...(field_ids); }

template<typename... field_ids>
int DummyGrid<field_ids...>::dummy_grid_function_b() { return m + sizeof...(field_ids); }


// -------------------------------------

template<typename field_set> struct InstantiateDummyGrid {};

template<typename... field_ids>
struct InstantiateDummyGrid< exanb::FieldSet<field_ids...> > 
{
  std::shared_ptr< DummyGrid<field_ids...> > m;
  InstantiateDummyGrid()
  {
    m->dummy_grid_function_a();
    m->dummy_grid_function_b();
  }
};

template<typename... field_sets>
struct InstantiateDummyGrids
{
  std::tuple< InstantiateDummyGrid<field_sets> ... > m;
};

template<typename field_sets> struct InstantiateDummyGridsHelper {};
template<typename... field_sets> struct InstantiateDummyGridsHelper< exanb::FieldSets<field_sets...> >
{
  InstantiateDummyGrids<field_sets...> m;
};

// automatic template instantiation does not work yet for classes, needs to be fixed
template class InstantiateDummyGridsHelper< exanb::StandardFieldSets >;
template class DummyGrid<exanb::field::_ep, exanb::field::_ax, exanb::field::_ay, exanb::field::_az, exanb::field::_vx, exanb::field::_vy, exanb::field::_vz>;
template class DummyGrid<exanb::field::_ep, exanb::field::_ax, exanb::field::_ay, exanb::field::_az, exanb::field::_vx, exanb::field::_vy, exanb::field::_vz, exanb::field::_id, exanb::field::_type,exanb::field::_virial, exanb::field::_idmol, exanb::field::_cmol, exanb::field::_charge>;
template class DummyGrid<exanb::field::_ep, exanb::field::_ax, exanb::field::_ay, exanb::field::_az, exanb::field::_vx, exanb::field::_vy, exanb::field::_vz, exanb::field::_id, exanb::field::_type>;
template class DummyGrid<exanb::field::_id, exanb::field::_sl>;
