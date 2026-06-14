/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#pragma once

#include <exanb/compute/field_combiners.h>
#include <exanb/compute/math_functors.h>
#include <exanb/core/particle_type_properties.h>
#include <exanb/core/grid.h>
#include <onika/flat_tuple.h>
#include <vector>
#include <span>

namespace exanb
{

  using GridAdditionalFieldsView = onika::FlatTuple<
    std::span< TypePropertyScalarCombiner > ,
    std::span< TypePropertyVec3Combiner > ,
    std::span< TypePropertyMat3Combiner > ,
    std::span< field::generic_real > ,
    std::span< field::generic_vec3 > ,
    std::span< field::generic_mat3 > >;

  struct GridAdditionalFields
  {
    std::vector< TypePropertyScalarCombiner > m_type_real_fields;
    std::vector< TypePropertyVec3Combiner > m_type_vec3_fields;
    std::vector< TypePropertyMat3Combiner > m_type_mat3_fields;
    std::vector< field::generic_real > m_opt_real_fields;
    std::vector< field::generic_vec3 > m_opt_vec3_fields;
    std::vector< field::generic_mat3 > m_opt_mat3_fields;

    template<typename GridT>
    inline GridAdditionalFields( GridT& grid , ParticleTypeProperties * optional_type_properties = nullptr )
    {
      // add per particle type scalar attributes
      if( optional_type_properties != nullptr )
      {
        for(const auto & it : optional_type_properties->m_scalars) m_type_real_fields.push_back( { make_type_property_functor( it.first , it.second.data() ) } );
        for(const auto & it : optional_type_properties->m_vectors) m_type_vec3_fields.push_back( { make_type_property_functor( it.first , it.second.data() ) } );
        // for(const auto & it : optional_type_properties->m_matrices) m_type_mat3_fields.push_back( { make_type_property_functor( it.first , it.second.data() ) } );
      }

      // add dynamic (grid optional flat arrays) particle scalars
      for(const auto & opt_name : grid->optional_scalar_fields()) m_opt_real_fields.push_back( field::mk_generic_real(opt_name) );
      for(const auto & opt_name : grid->optional_vec3_fields()  ) m_opt_vec3_fields.push_back( field::mk_generic_vec3(opt_name) );
      for(const auto & opt_name : grid->optional_mat3_fields()  ) m_opt_mat3_fields.push_back( field::mk_generic_mat3(opt_name) );
    }

    inline GridAdditionalFieldsView view()
    {
      return { m_type_real_fields, m_type_vec3_fields, m_type_mat3_fields, m_opt_real_fields, m_opt_vec3_fields , m_opt_mat3_fields };
    }
  };


  template<class TransformFuncT>
  struct FieldTransformer
  {
    TransformFuncT m_transform;
    std::string_view m_suffix;
  };

  template<class TransformersTupleT,class FieldT>
  struct ApplyOnParticleField
  {

    template<class FuncT, class TransformFuncT>
    static inline void apply_transformer(const FuncT& func, const FieldT& f, const FieldTransformer<TransformFuncT>& transform)
    {
      if constexpr ( onika::lambda_is_callable_with_args_v<TransformFuncT,typename FieldT::value_type> )
      {
        TransformCellParticleFieldAccessor<TransformFuncT,FieldT> tfield = { transform.m_transform, f };
        tfield.transform_name( transform.m_suffix );
        func( tfield );
      }
    }

    template<class FuncT, size_t... I>
    static inline void apply_transformers(const FuncT& func, const FieldT& f, const TransformersTupleT& transformers, std::index_sequence<I...> )
    {
      ( ... , ( apply_transformer(func,f,transformers.get(onika::tuple_index<I>)) ) );
    }

    template<class FuncT>
    static inline void apply(const FuncT& func, const FieldT& f, const TransformersTupleT& transformers)
    {
      func(f);
      constexpr size_t NTransformers = transformers.size();
      apply_transformers(func,f,transformers,std::make_index_sequence<NTransformers>{} );
    }
  };

  template<class TransformersTupleT,class FieldT>
  struct ApplyOnParticleField<TransformersTupleT, std::span<FieldT> >
  {
    template<class FuncT>
    static inline void apply(const FuncT& func, const std::span<FieldT>& fvec, const TransformersTupleT& transformers)
    {
      for(const auto& f : fvec) ApplyOnParticleField<TransformersTupleT,FieldT>::apply(func,f,transformers);
    }
  };
  template<class TransformersTupleT,class... FieldsOrSpansT>
  struct ApplyOnParticleField<TransformersTupleT, onika::FlatTuple<FieldsOrSpansT...> >
  {
    template<class FuncT, size_t... I>
    static inline void apply( const FuncT& func, const onika::FlatTuple<FieldsOrSpansT...>& ftpl, const TransformersTupleT& transformers, std::index_sequence<I...> )
    {
      ( ... , ( ApplyOnParticleField<TransformersTupleT,FieldsOrSpansT>::apply(func,ftpl.get(onika::tuple_index<I>),transformers) ) );
    }
    template<class FuncT>
    static inline void apply(const FuncT& func, const onika::FlatTuple<FieldsOrSpansT...>& ftpl, const TransformersTupleT& transformers)
    {
      apply( func, ftpl, transformers, std::make_index_sequence<sizeof...(FieldsOrSpansT)>{} );
    }
  };

  template<class FuncT, class FieldT, class TransformersTupleT = onika::FlatTuple<> >
  inline void apply_on_particle_field( const FuncT& func, const FieldT& f , const TransformersTupleT& transformers = {} )
  {
    ApplyOnParticleField<TransformersTupleT,FieldT>::apply(func,f,transformers);
  }

}
