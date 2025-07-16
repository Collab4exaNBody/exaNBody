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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <onika/memory/allocator.h>

#include <onika/soatl/field_pointer_tuple.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/compute/generic_vec3_field_functors.h>

namespace exanb
{

  template<
    class GridT,
    class Field_X, class Field_Y, class Field_Z,
    class OpT ,
    class = AssertGridHasFields< GridT, Field_X, Field_Y, Field_Z >
    >
  class GenericVec3Operator : public OperatorNode
  {  
    ADD_SLOT( Vec3d , value , INPUT , Vec3d{0.,0.,0.} );
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT);

    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL );

    using compute_field_set_t = FieldSet< Field_X, Field_Y, Field_Z >;
    using has_field_id_t     = typename GridT:: template HasField <field::_id>;
    static constexpr bool has_field_id = has_field_id_t::value;
    static constexpr bool has_separate_r_fields = ! ( std::is_same_v<Field_X,field::_rx> && std::is_same_v<Field_Y,field::_ry> && std::is_same_v<Field_Z,field::_rz> );
    
    using compute_field_set_region_t = 
      std::conditional_t< has_field_id ,      
        std::conditional_t< has_separate_r_fields ,
          FieldSet< field::_rx , field::_ry , field::_rz , field::_id, Field_X, Field_Y, Field_Z >
          ,
          FieldSet< field::_id, Field_X, Field_Y, Field_Z > >
        ,
        std::conditional_t< has_separate_r_fields ,
          FieldSet< field::_rx , field::_ry , field::_rz , Field_X, Field_Y, Field_Z >
          ,
          FieldSet< Field_X, Field_Y, Field_Z > > 
        >;

  public:
    inline void execute () override final
    {
      ldbg<<"GenericVec3Operator: value="<<(*value)<<std::endl;

      if( grid->number_of_cells() == 0 ) return;

      if( region.has_value() )
      {
        if( !particle_regions.has_value() )
        {
          fatal_error() << "GenericVec3Operator: region is defined, but particle_regions has no value" << std::endl;
        }        
        if( region->m_nb_operands==0 )
        {
          ldbg << "rebuild CSG from expr "<< region->m_user_expr << std::endl;
          region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
        }
        ParticleRegionCSGShallowCopy prcsg = *region;
        ldbg << "Particle Region CSG\n\tm_expr = "<<prcsg.m_expr<<std::endl
             << "\tm_nb_regions = " << static_cast<int>(prcsg.m_nb_regions)<<std::endl
             << "\tm_nb_operands = " << static_cast<int>(prcsg.m_nb_operands)<<std::endl
             << "\tm_nb_operands_log2 = " << static_cast<int>(prcsg.m_nb_operands_log2)<<std::endl
             << "\tm_operand_places =";
        for(unsigned int i=0;i<prcsg.m_nb_operands;i++) ldbg<<" "<<static_cast<int>(prcsg.m_operand_places[i]);
        ldbg << std::endl << "\tm_regions = " << std::endl;
        for(unsigned int i=0;i<prcsg.m_nb_regions;i++)
        {
          const auto & R = prcsg.m_regions[i];
          ldbg << "\t\t"<< R.m_name << std::endl;
          ldbg << "\t\t\tQuadric = "<< R.m_quadric << std::endl;
          ldbg << "\t\t\tBounds  = "<< R.m_bounds << std::endl;
          ldbg << "\t\t\tId range = [ "<< R.m_id_start << " ; "<<R.m_id_end<<" [" << std::endl;
        }

        field_accessor_tuple_from_field_set_t<compute_field_set_region_t> cp_fields = {};
        GenericVec3RegionFunctor<OpT> func = { prcsg , { *value  } };
        compute_cell_particles( *grid , false , func , cp_fields , parallel_execution_context() );            
      }
      else
      {
        field_accessor_tuple_from_field_set_t<compute_field_set_t> cp_fields = {};
        GenericVec3Functor<OpT> func = { { *value } };
        compute_cell_particles( *grid , false , func , cp_fields , parallel_execution_context() );            
      }
      
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsSequence() )
      {
        tmp["value"] = node;
      }
      else if( node.IsScalar() )
      {
        double x = node.as<onika::physics::Quantity>().convert();
        tmp["value"] = std::vector<double> { x , x , x };
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };

  template<class GridT> using ShiftPositionOperator = GenericVec3Operator< GridT, field::_rx,field::_ry,field::_rz , InPlaceAddFunctor<onika::math::Vec3d> >;
  template<class GridT> using ShiftVelocityOperator = GenericVec3Operator< GridT, field::_vx,field::_vy,field::_vz , InPlaceAddFunctor<onika::math::Vec3d> >;
  template<class GridT> using ScalePositionOperator = GenericVec3Operator< GridT, field::_rx,field::_ry,field::_rz , InPlaceMulFunctor<onika::math::Vec3d> >;
  template<class GridT> using ScaleVelocityOperator = GenericVec3Operator< GridT, field::_vx,field::_vy,field::_vz , InPlaceMulFunctor<onika::math::Vec3d> >;
  template<class GridT> using SetVelocityOperator   = GenericVec3Operator< GridT, field::_vx,field::_vy,field::_vz , SetFirstArgFunctor<onika::math::Vec3d> >;
  template<class GridT> using SetForceOperator      = GenericVec3Operator< GridT, field::_fx,field::_fy,field::_fz , SetFirstArgFunctor<onika::math::Vec3d> >;
  
 // === register factories ===  
  ONIKA_AUTORUN_INIT(generic_op_vec3)
  {
   OperatorNodeFactory::instance()->register_factory( "shift_r", make_grid_variant_operator< ShiftPositionOperator > );
   OperatorNodeFactory::instance()->register_factory( "shift_v", make_grid_variant_operator< ShiftVelocityOperator > );
   OperatorNodeFactory::instance()->register_factory( "scale_r", make_grid_variant_operator< ScalePositionOperator > );
   OperatorNodeFactory::instance()->register_factory( "scale_v", make_grid_variant_operator< ScaleVelocityOperator > );
   OperatorNodeFactory::instance()->register_factory( "set_velocity", make_grid_variant_operator< SetVelocityOperator > );
   OperatorNodeFactory::instance()->register_factory( "set_force", make_grid_variant_operator< SetForceOperator > );
  }

}

