#pragma once

#include <onika/soatl/field_tuple.h>
#include <exanb/field_sets.h>
#include <onika/cuda/cuda.h>

namespace exanb
{
  

  namespace UpdateGhostsUtils
  {
    template<typename FieldSetT> struct FieldSetToParticleTuple;
    template<typename... field_ids> struct FieldSetToParticleTuple< FieldSet<field_ids...> > { using type = onika::soatl::FieldTuple<field_ids...>; };
    template<typename FieldSetT> using field_set_to_particle_tuple_t = typename FieldSetToParticleTuple<FieldSetT>::type;

    template<typename TupleT, class fid, class T>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_field_shift( TupleT& t , onika::soatl::FieldId<fid> f, T x )
    {
      static constexpr bool has_field = onika::soatl::field_tuple_has_field_v<TupleT,fid>;
      if constexpr ( has_field ) { t[ f ] += x; }
    }
    
    template<typename TupleT>
    ONIKA_HOST_DEVICE_FUNC
    static inline void apply_r_shift( TupleT& t , double x, double y, double z )
    {
      apply_field_shift( t , field::rx , x );
      apply_field_shift( t , field::ry , y );
      apply_field_shift( t , field::rz , z );

      if constexpr ( HAS_POSITION_BACKUP_FIELDS )
      {
        apply_field_shift( t , PositionBackupFieldX , x );
        apply_field_shift( t , PositionBackupFieldY , y );
        apply_field_shift( t , PositionBackupFieldZ , z );
      }
    }
    
  } // template utilities used only inside UpdateGhostsNode

}

