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
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/domain.h>

#include <onika/math/basic_types_yaml.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
//#include "exanb/vector_utils.h"
#include <exanb/core/check_particles_inside_cell.h>
#include <onika/physics/constants.h>
#include <onika/parallel/random.h>
#include <onika/thread.h>
#include <exanb/core/particle_type_id.h>

#include <exanb/grid_cell_particles/particle_localized_filter.h>
#include <exanb/grid_cell_particles/generate_particle_lattice.h>

#include <mpi.h>
#include <string>

namespace exanb
{

  template< class GridT
          , class ParticleTypeField
          >
  class RegionLattice : public OperatorNode
  {
    using StringVector = std::vector<std::string>;
    using Vec3dVector = std::vector<Vec3d>;

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( MPI_Comm        , mpi          , INPUT , MPI_COMM_WORLD  );
    ADD_SLOT( ReadBoundsSelectionMode, bounds_mode   , INPUT , ReadBoundsSelectionMode::FILE_BOUNDS );
    ADD_SLOT( Domain          , domain       , INPUT_OUTPUT );
    ADD_SLOT( GridT           , grid         , INPUT_OUTPUT );

    // get a type id from a type name
    ADD_SLOT( ParticleTypeMap , particle_type_map , INPUT , OPTIONAL );

    // limit lattice generation to specified region
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT_OUTPUT , OPTIONAL );

    // limit lattice generation to places where some mask has some value
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT , OPTIONAL );
    ADD_SLOT( std::string    , grid_cell_mask_name , INPUT , OPTIONAL );
    ADD_SLOT( double         , grid_cell_mask_value , INPUT , OPTIONAL );
    
    // limit lattice generation given a source term spatial function
    ADD_SLOT( ScalarSourceTermInstance , user_function , INPUT , OPTIONAL , DocString{"user scalar source term function defining space locations where particle generation is enabled"} );
    ADD_SLOT( double                   , user_threshold, INPUT , 0.0 , DocString{"if user_function(...) returns a value greater or equal to this threshold, allows particle generation, otherwise prevent it"} );

    // Variables related to the crystal structure : can be a predefined structure or a custom one
    ADD_SLOT( std::string      , structure    , INPUT , REQUIRED );
    ADD_SLOT( StringVector     , types        , INPUT , REQUIRED );    
    ADD_SLOT( Vec3d            , size         , INPUT , REQUIRED );    
    ADD_SLOT( ssize_t          , np           , INPUT , OPTIONAL );    
    ADD_SLOT( Vec3dVector      , positions    , INPUT , OPTIONAL );    
    ADD_SLOT( double           , noise        , INPUT , 0.0);
    ADD_SLOT( double           , noise_cutoff , INPUT , OPTIONAL );
    ADD_SLOT( Vec3d            , shift        , INPUT , Vec3d{0.0,0.0,0.0} );

    // Variables related to the special geometry, here a cylinder inside/outside which we keep/remove the particles. WARNING : be careful with the PBC    
    ADD_SLOT( std::string      , void_mode          , INPUT , "none"); // none means no void, simple is the one void mode, porosity means randomly distributed voids
    ADD_SLOT( Vec3d            , void_center        , INPUT , Vec3d{0., 0., 0.});
    ADD_SLOT( double           , void_radius        , INPUT , 0.);
    ADD_SLOT( double           , void_porosity      , INPUT , 0.);
    ADD_SLOT( double           , void_mean_diameter , INPUT , 0.);        
    
  public:
    inline void execute () override final
    {
      // Checking the required lattice :
      //   if 'CUSTOM' -> check required fields then create LatticeCollection
      //   if one of predefined lattices 'SC' 'BCC' 'FCC' 'HCP' etc... -> check required fields then create Lattice Collection

      LatticeCollection lattice;
      lattice.m_structure = *structure;
      lattice.m_size = *size;
      if (*structure == "CUSTOM") {
        assert( np.has_value() );
        assert( positions.has_value() );
        lattice.m_np = *np;
        lattice.m_types = *types;
        // Checking that positions are contained between 0 and 1 -> fractional coordinates
#       ifndef NDEBUG
        for (size_t i=0;i<positions->size();i++) {
          double px = positions->at(i).x;
          double py = positions->at(i).y;
          double pz = positions->at(i).z;
          assert( px >=0. && px <= 1. );
          assert( py >=0. && py <= 1. );
          assert( pz >=0. && pz <= 1. );
        }
#       endif
        lattice.m_positions = *positions;
      } else if (*structure == "SC") {
        lattice.m_np = 1;
        lattice.m_types = *types;
        lattice.m_positions = { {.5, .5, .5 } };
      } else if (*structure == "BCC") {
        lattice.m_np = 2;
        lattice.m_types = *types;
        lattice.m_positions = { {.0, .0, .0} ,
                                {.5, .5, .5} };
      } else if (*structure == "2BCT") {
        lattice.m_np = 4;
        lattice.m_types = *types;
        lattice.m_positions = { {.0, .0, .0 } ,
                                {.0, .5, .25} ,
                                {.5, .5, .5 } ,
                                {.5, .0, .75} };
      } else if (*structure == "FCC") {
        lattice.m_np = 4;
        lattice.m_types = *types;
        lattice.m_positions = { {.0, .0, .0} ,
                                {.0, .5, .5} ,
                                {.5, .0, .5} ,
                                {.5, .5, .0} };	
      } else if (*structure == "HCP") {
        lattice.m_np = 4;
        lattice.m_types = *types;
        lattice.m_positions = { {0.25,    0.25000000,    0.25} ,
                                {0.75,    0.75000000,    0.25} ,
                                {0.25,    0.58333333,    0.75} ,
                                {0.75,    0.08333333,    0.75} };
        lattice.m_size.y *= (2. * sin(120. * M_PI / 180.));
      } else if (*structure == "c-DIA") {
        lattice.m_np = 8;
        lattice.m_types = *types;
        lattice.m_positions = { {.00, .00, .00} ,
                                {.50, .50, .00} ,
                                {.00, .50, .50} ,
                                {.50, .00, .50} ,
                                {.25, .25, .25} ,
                                {.75, .75, .25} ,          
                                {.75, .25, .75} ,
                                {.25, .75, .75} };
      } else if (*structure == "h-DIA") {
        lattice.m_np = 8;
        lattice.m_types = *types;
        lattice.m_positions = { {0.50000000,    0.16666667,    0.50000000} ,
                                {0.00000000,    0.66666667,    0.50000000} ,
                                {0.50000000,    0.16666667,    0.87500000} ,
                                {0.00000000,    0.66666667,    0.87500000} ,
                                {0.00000000,    0.33333333,    0.00000000} ,
                                {0.50000000,    0.83333333,    0.00000000} ,
                                {0.00000000,    0.33333333,    0.37500000} ,
                                {0.50000000,    0.83333333,    0.37500000} };
        lattice.m_size.y *= (2. * sin(120. * M_PI / 180.));
      } else if (*structure == "graphite") {
        lattice.m_np = 8;
        lattice.m_types = *types;
        lattice.m_positions = { {0.0,    0.00000000,    0.5} ,
                                {0.5,    0.50000000,    0.5} ,
                                {0.5,    0.16666667,    0.5} ,
                                {0.0,    0.66666667,    0.5} ,
                                {0.0,    0.33333333,    0.0} ,
                                {0.5,    0.83333333,    0.0} ,
                                {0.0,    0.00000000,    0.0} ,
                                {0.5,    0.50000000,    0.0} };
        lattice.m_size.y *= (2. * sin(120. * M_PI / 180.));
      }
      
      const double noise_cutoff_ifset = noise_cutoff.has_value() ? *noise_cutoff : -1.0;
      std::shared_ptr<exanb::ScalarSourceTerm> user_source_term = nullptr;
      if( user_function.has_value() ) user_source_term = *user_function;
      
      ParticleTypeMap mock_particle_type_map;
      mock_particle_type_map.clear();
      if( particle_type_map.has_value() ) mock_particle_type_map = *particle_type_map;
      
      generate_particle_lattice( *mpi, *bounds_mode, *domain, *grid, mock_particle_type_map, particle_regions.get_pointer(), region.get_pointer()
                               , grid_cell_values.get_pointer(), grid_cell_mask_name.get_pointer(), grid_cell_mask_value.get_pointer(), user_source_term, *user_threshold
                                 , lattice, *noise, noise_cutoff_ifset, *shift
                               , *void_mode, *void_center, *void_radius, *void_porosity, *void_mean_diameter, ParticleTypeField{} );
    }
    
  };

}
