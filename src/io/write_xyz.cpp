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

#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/io/write_xyz.h>
#include <exanb/core/grid_additional_fields.h>

namespace exanb
{

  template<class GridT>
  class WriteXYZGeneric : public OperatorNode
  {    
    using StringList = std::vector<std::string>;
    using StringMap = std::map<std::string,std::string>;
        
    ADD_SLOT( MPI_Comm        , mpi      , INPUT );
    ADD_SLOT( GridT           , grid     , INPUT );
    ADD_SLOT( Domain          , domain   , INPUT );
    ADD_SLOT( bool            , ghost    , INPUT , false );
    ADD_SLOT( std::string     , filename , INPUT , "output"); // default value for backward compatibility
    ADD_SLOT( StringList      , fields   , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to write"} );
    ADD_SLOT( StringMap       , units    , INPUT , StringMap( { {"position","m"} , {"velocity","m/s"} , {"force","m/s/kg"} } ) , DocString{"Units to be used for specific fields."} );
    ADD_SLOT( StringMap       , field_alias , INPUT , StringMap( { {"position","pos"} , {"velocity","vel"} } ) , DocString{"Optional field renaming for shorter names."} );
    ADD_SLOT( ParticleTypeProperties , particle_type_properties , INPUT , ParticleTypeProperties{} );
    ADD_SLOT( double          , physical_time  , INPUT , 0.0 );
    
    template<class... fid>
    inline void execute_on_field_set( FieldSet<fid...> ) 
    {
      using has_field_type_t = typename GridT:: template HasField < field::_type >;
      static constexpr bool has_field_type = has_field_type_t::value;

      int rank=0;
      MPI_Comm_rank(*mpi, &rank);

      std::unordered_map<std::string,double> conv_scale;
      for(const auto& umap : *units)
      {
        const auto s = "1.0 " + umap.second;
        bool conv_ok = false;
        auto q = onika::physics::quantity_from_string( s , conv_ok );
        if( ! conv_ok ) { fatal_error() << "Failed to parse unit string '"<<s<<"'"<<std::endl; }
        conv_scale[umap.first] = q.convert();
      }

      // property name for position must be 'Position'
      StringList flist = { "position" };
      for(const auto& f : *fields) { if( f != "position" ) flist.push_back(f); }

      // formatter helps writing correct type names, renamed field and convert final values to desired units
      write_xyz_details::DefaultFieldFormatter field_formatter = { *units , conv_scale , *field_alias };

      const auto& tp = *particle_type_properties;
      auto particle_type_func = [&tp](auto cells, size_t c, size_t pos) -> const char *
      {
        if constexpr ( has_field_type )
        {
          return tp.m_names[ cells[c][field::type][pos] ].c_str();
        }
        return "XX";
      };

      // generated particle fields
      ProcessorRankCombiner processor_id = { {rank} };
      PositionVec3Combiner position = {};
      VelocityVec3Combiner velocity = {};
      ForceVec3Combiner    force    = {};

      // optional fields
      ParticleTypeProperties * optional_type_properties = nullptr;
      if ( has_field_type && particle_type_properties.has_value() ) optional_type_properties = particle_type_properties.get_pointer();
      GridAdditionalFields add_fields( grid , optional_type_properties );
      auto [ type_real_fields, type_vec3_fields, type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields ] = add_fields.view();

      write_xyz_details::write_xyz_grid_fields( ldbg, *mpi, *grid, *domain, flist, *filename, particle_type_func, field_formatter, *ghost, *physical_time
                                              , position, velocity, force, processor_id
                                              , type_real_fields, type_vec3_fields, type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields
                                              , onika::soatl::FieldId<fid>{} ... );
    }

    public:
    inline void execute() override
    {
      using GridFieldSet = RemoveFields< typename GridT::field_set_t , FieldSet< field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz> >;
      execute_on_field_set( GridFieldSet{} );
    }
    
  };


  // === register factories ===  
  ONIKA_AUTORUN_INIT(write_xyz)
  {
    OperatorNodeFactory::instance()->register_factory( "write_xyz", make_grid_variant_operator< WriteXYZGeneric > );
  }

}
