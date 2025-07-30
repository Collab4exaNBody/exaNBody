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
#include <exanb/core/simple_block_rcb.h>

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
  class BulkLattice : public OperatorNode
  {
    using StringVector = std::vector<std::string>;
    using Vec3dVector = std::vector<Vec3d>;

    // -----------------------------------------------
    // Operator slots
    // -----------------------------------------------    
    ADD_SLOT( MPI_Comm        , mpi          , INPUT , MPI_COMM_WORLD  );
    ADD_SLOT( Domain          , domain       , INPUT_OUTPUT );
    ADD_SLOT( GridT           , grid         , INPUT_OUTPUT );

    // get a type id from a type name
    ADD_SLOT( ParticleTypeMap , particle_type_map , INPUT , OPTIONAL );

    // limit lattice generation to specified region
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT_OUTPUT , OPTIONAL , DocString{"Region identifier or boolean expression of region identifiers to be filled with particles."});

    // limit lattice generation to places where some mask has some value
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT , OPTIONAL );
    ADD_SLOT( std::string    , grid_cell_mask_name , INPUT , OPTIONAL , DocString{"(YAML: string) Name of grid_cell_values mask that will act as a mask region in which the gaussian noise should be applied."} );
    ADD_SLOT( double         , grid_cell_mask_value , INPUT , OPTIONAL , DocString{"(YAML: float) Value of the grid_cell_values used to define the mask."} );
    
    // limit lattice generation given a source term spatial function
    ADD_SLOT( ScalarSourceTermInstance , user_function , INPUT , OPTIONAL , DocString{"user scalar source term function defining space locations where particle generation is enabled"} );
    ADD_SLOT( double                   , user_threshold, INPUT , 0.0 , DocString{"if user_function(...) returns a value greater or equal to this threshold, allows particle generation, otherwise prevent it"} );

    // Variables related to the crystal structure : can be a predefined structure or a custom one
    ADD_SLOT( std::string      , structure    , INPUT , REQUIRED , DocString{"(YAML: string) Name of required crystal structure. SC or BCC or FCC or 2BCT or HCP or h-DIA or c-DIA or graphite."} );
    ADD_SLOT( StringVector     , types        , INPUT , REQUIRED , DocString{"(YAML: list) List of types. Must be consistent with the required crystal structure. (SC:1 | BCC:2 | FCC,2BCT,HCP:4 | c-DIA,h-DIA,graphite:8)"} );
    ADD_SLOT( Vec3d            , size         , INPUT , REQUIRED , DocString{"(YAML: Vec3d) Vector with crystal unit cell lengths."} );
    ADD_SLOT( Vec3dVector      , positions    , INPUT , OPTIONAL , DocString{"(YAML: list of Vec3d) Only when structure CUSTOM is required. List of positions of individual atoms. Must be consistent with number of types provided."} );
    ADD_SLOT( bool             , deterministic_noise , INPUT , false );    
    ADD_SLOT( Vec3d            , shift        , INPUT , Vec3d{0.0,0.0,0.0} , DocString{"(YAML: Vec3d) Vector for shifting atomic positions in the unit cell."} );

    // this additional parameter will decide domain's dimensions
    ADD_SLOT( IJK             , repeat        , INPUT , IJK{1,1,1} , DocString{"(YAML: Vec3d) Vector with required replication number of unit cell along x,y,z."} );

    // Variables related to the special geometry, here a cylinder inside/outside which we keep/remove the particles. WARNING : be careful with the PBC    
    ADD_SLOT( std::string      , void_mode          , INPUT , "none", DocString{"(YAML: string) Void mode (none, simple or porosity)."} );
    ADD_SLOT( Vec3d            , void_center        , INPUT , Vec3d{0., 0., 0.}, DocString{"(YAML: Vec3d) Void center position in the case of a simple void is required."} );
    ADD_SLOT( double           , void_radius        , INPUT , 0., DocString{"(YAML: float) Void radius in the case of a simple void is requied."} );
    ADD_SLOT( double           , void_porosity      , INPUT , 0., DocString{"(YAML: float) Target porosity in the case the void mode is set to porosity."} );
    ADD_SLOT( double           , void_mean_diameter , INPUT , 0., DocString{"(YAML: float) Mean void diameterin the case the void mode is set to porosity."} );
    
  public:
    inline void execute () override final
    {
      // Checking the required lattice :
      //   if 'CUSTOM' -> check required fields then create LatticeCollection
      //   if one of predefined lattices 'SC' 'BCC' 'FCC' 'HCP' etc... -> check required fields then create Lattice Collection

      if( positions.has_value() )
      {
        if( positions->size() != types->size() )
        {
          fatal_error() << "positions is defined with a different number of elements ("<<positions->size()<<") than types ("<<types->size()<<")"<<std::endl;
        }
      }

      if( (*structure == "SC") && (types->size() != 1) )
        {
          fatal_error() << "For SC structure, types must contain 1 specy. Example: types: [ A ]"<<std::endl;
        }
      else if ( (*structure == "BCC") && (types->size() != 2) )
        {
          fatal_error() << "For BCC structure, types must contain 2 species. Example: types: [ A, A ]"<<std::endl;
        }
      else if ( (*structure == "2BCT" || *structure == "FCC" || *structure == "HCP") && (types->size() != 4) )
        {
          fatal_error() << "For 2BCT, FCC or HCP structure, types must contain 4 species. Example: types: [ A, A, B, B ]"<<std::endl;
        }
      else if ( (*structure == "c-DIA" || *structure == "h-DIA" || *structure == "graphite") && (types->size() != 8) )
        {
          fatal_error() << "For c-DIA, h-DIA or graphite structure, types must contain 8 species. Example: types: [ A, A, B, B, C, C, D, D ]"<<std::endl;
        }
      
      LatticeCollection lattice;
      lattice.m_structure = *structure;
      lattice.m_size = *size;
      if (*structure == "CUSTOM") {
        if( ! positions.has_value() )
        {
          fatal_error() << "CUSTOM structure is selected, but 'positions' is not defined"<<std::endl;
        }
        lattice.m_np = types->size();
        lattice.m_types = *types;
        // Checking that positions are contained between 0 and 1 -> fractional coordinates
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

      for(size_t i=0;i<lattice.m_positions.size();i++)
      {
        const auto p = lattice.m_positions[i];
        if( p.x < 0.0 || p.x > 1. || p.y < 0.0 || p.y > 1. || p.z < 0.0 || p.z > 1. )
        {
          fatal_error()<<"lattice position #"<<i<<", with coordinates "<<p<<", is out of unit cell"<<std::endl;
        }
      }

      if( lattice.m_types.size() != size_t(lattice.m_np) ) { fatal_error()<<lattice.m_types.size()<<"types are defined, but structure "<< (*structure) <<" requires exactly "<<lattice.m_np<<std::endl; }
      
      std::shared_ptr<exanb::ScalarSourceTerm> user_source_term = nullptr;
      if( user_function.has_value() ) user_source_term = *user_function;
      
      // domain setup in order to ensure the commensurability
      // between the desired lattice and the domain size
      
      domain->set_expandable( false );
      domain->set_periodic_boundary( true , true , true );
      domain->set_xform( make_identity_matrix() );
      IJK ncells = { std::floor(repeat->i * lattice.m_size.x / domain->cell_size()) , std::floor(repeat->j * lattice.m_size.y / domain->cell_size()) , std::floor(repeat->k * lattice.m_size.z / domain->cell_size()) };
      domain->set_grid_dimension( ncells );      
      Vec3d xformdiag = { (repeat->i * lattice.m_size.x) / (ncells.i * domain->cell_size()), (repeat->j * lattice.m_size.y) / (ncells.j * domain->cell_size()), (repeat->k * lattice.m_size.z) / (ncells.k * domain->cell_size()) };
      domain->set_xform( make_diagonal_matrix( xformdiag ) );
      domain->set_bounds( { Vec3d{0.,0.,0.} , Vec3d{ncells.i * domain->cell_size(), ncells.j * domain->cell_size(), ncells.k * domain->cell_size()} } );

      // include init rcb grid here without checking domain as in basic lattice_generator
      // because bulk_generator intializes it on its own
      
      // MPI Initialization
      int rank=0, np=1;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &np);

      // compute local processor's grid size and location so that cells are evenly distributed
      GridBlock in_block = { IJK{0,0,0} , domain->grid_dimension() };
      ldbg<<"In  block = "<< in_block << std::endl;
      GridBlock out_block = simple_block_rcb( in_block, np, rank );
      ldbg<<"Out block = "<< out_block << std::endl;

      // initializes local processor's grid
      grid->set_offset( out_block.start );
      grid->set_origin( domain->bounds().bmin );
      grid->set_cell_size( domain->cell_size() );
      IJK local_grid_dim = out_block.end - out_block.start;
      grid->set_dimension( local_grid_dim );
      grid->rebuild_particle_offsets();      

      ParticleTypeMap mock_particle_type_map;
      mock_particle_type_map.clear();
      if( particle_type_map.has_value() ) mock_particle_type_map = *particle_type_map;
      
      generate_particle_lattice( ldbg, *mpi, *domain, *grid, mock_particle_type_map, particle_regions.get_pointer(), region.get_pointer()
                                 , grid_cell_values.get_pointer(), grid_cell_mask_name.get_pointer(), grid_cell_mask_value.get_pointer(), user_source_term, *user_threshold
                                 , lattice, *shift, *deterministic_noise
                                 , *void_mode, *void_center, *void_radius, *void_porosity, *void_mean_diameter, ParticleTypeField{} );
    }

    inline std::string documentation() const override final
    {
      return R"EOF(

This operator replicates a unit cell with particles as specific locations and automatically define the simulation domain accordingly. Various crystal structures can be required: SC, BCC, 2BCT, FCC, HCP, h-DIA, c-DIA, graphite and CUSTOM. These unit cells are all orthorhombic (all angles = 90 deg, but lengths can be different). In the case of CUSTOM, the user needs to provide the particles location as well.

This operator requires the domain operator to be called before but only with the cell_size attributes. Then, both the domain initialization and the init_rcb_grid operator are done internally. Beware, this operator is designed to generate perfectly commensurate simulation domain with 3D periodic boundary conditions.

Usage examples:

input_data:
  - domain:
      cell_size: 5.0 ang
  - bulk_lattice:
      structure: BCC
      types: [ Ta , Ta]
      size: [ 3. ang , 3. ang , 3. ang ]
      repeats: [ 20, 30, 45 ]

)EOF";
    } 
  };

}
