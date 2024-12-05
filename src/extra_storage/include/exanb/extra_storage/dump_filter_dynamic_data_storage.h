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

#include <onika/soatl/field_tuple.h>
#include <exanb/io/sim_dump_io.h>
#include <iostream>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/quaternion_operators.h>
#include <onika/math/quaternion_stream.h>
#include <exanb/extra_storage/dynamic_data_storage.hpp>
#include <exanb/extra_storage/reader_dynamic_data_storage.hpp>


namespace exanb
{
  template< class GridT , class ItemType, class DumpFieldSet >
  struct ParticleDumpFilterWithExtraDataStorage
  {
    using GridFieldSet = typename GridT::Fields;
    using TupleT = onika::soatl::FieldTupleFromFieldIds< DumpFieldSet >;
    using StorageType = TupleT; 

    GridExtraDynamicDataStorageT<ItemType>& grid_item;
    GridT& grid;
    ExtraDynamicDataStorageReadHelper<ItemType> extra_data_read_helper;
    double scale_cell_size = 1.0;
    bool enable_extra_data = true;

    // optional modification of domain periodicity
    bool override_periodicity = false;
    bool periodic_x = false;
    bool periodic_y = false;
    bool periodic_z = false;

    // optional override of domain expandability
    bool override_expandable = false;
    bool expandable = false;

    // optionally override domain bounds
    bool override_domain_bounds = false;
    bool shrink_to_fit = false;
    AABB domain_bounds = { {0,0,0} , {0,0,0} };

    // for forward compatibility with dump_reader_allow_initial_position_xform branch
    inline void process_domain(Domain& domain , Mat3d& particle_read_xform)
    {
      particle_read_xform = make_identity_matrix();
      if( scale_cell_size != 1.0 )
      {
        const Vec3d dom_size = domain.bounds_size();
        const Mat3d dom_xform = domain.xform();

        double desired_cell_size = domain.cell_size() * scale_cell_size;
        IJK grid_dims = make_ijk( Vec3d{0.5,0.5,0.5} + ( dom_size / desired_cell_size ) ); // round to nearest
        domain.set_grid_dimension( grid_dims );
        domain.set_cell_size( desired_cell_size );
        // domain bounds should remain the same
        domain.set_bounds( { domain.origin() , domain.origin() + ( grid_dims * desired_cell_size ) } );
        const Vec3d dom_new_size = domain.bounds_size();

        particle_read_xform = diag_matrix( dom_new_size / dom_size );
        const Mat3d dom_new_xform = inverse( particle_read_xform ) * dom_xform;
        domain.set_xform( dom_new_xform );
      }
      if( override_periodicity )
      {
        domain.set_periodic_boundary( periodic_x , periodic_y , periodic_z );
      }
      if( override_expandable )
      {
        domain.set_expandable( expandable );
      }

			// domain_bounds = domain.bounds(); // looks wrong

      if( override_domain_bounds )
      {
        if( ! domain.xform_is_identity() )
        {
          if( ! is_diagonal( domain.xform() ) )
          {
            fatal_error() << "cannot force domain bounds on a domain with non diagonal transform matrix" << std::endl;
          }
          else
          {
            Vec3d diag = { domain.xform().m11 , domain.xform().m22 , domain.xform().m33 };
            domain_bounds.bmin = domain_bounds.bmin / diag;
            domain_bounds.bmax = domain_bounds.bmax / diag;
          }
        }
        if( domain.periodic_boundary_x() ) { domain_bounds.bmin.x = domain.bounds().bmin.x; domain_bounds.bmax.x = domain.bounds().bmax.x; }
        if( domain.periodic_boundary_y() ) { domain_bounds.bmin.y = domain.bounds().bmin.y; domain_bounds.bmax.y = domain.bounds().bmax.y; }
        if( domain.periodic_boundary_z() ) { domain_bounds.bmin.z = domain.bounds().bmin.z; domain_bounds.bmax.z = domain.bounds().bmax.z; }
      }
    }

    inline bool particle_input_filter(const Vec3d& r)
    {
      return ( ! override_domain_bounds ) || ( is_inside(domain_bounds,r) );
    }

    inline void post_process_domain(Domain& domain)
    {
      if( override_domain_bounds && shrink_to_fit)
      {
        const IJK cell_shift = make_ijk( floor( ( domain_bounds.bmin - domain.origin() ) / domain.cell_size() ) );
        const Vec3d new_origin = domain.origin() + cell_shift * domain.cell_size();
        const IJK new_grid_dims = make_ijk( ceil( ( domain_bounds.bmax - new_origin ) / domain.cell_size() ) );
        const Vec3d new_extent = new_origin + new_grid_dims * domain.cell_size();
        domain.set_bounds( { new_origin , new_extent } );
        domain.set_grid_dimension( new_grid_dims );
      }
    }

    inline void update_sats( const StorageType & ) { }
    inline void initialize_write() { }
    inline void finalize_write() { }

    inline void initialize_read()
    {
      extra_data_read_helper.initialize( grid.number_of_cells() );
    }

    inline void finalize_read()
    {
      if( enable_extra_data && ! extra_data_read_helper.m_out_item.empty() )
      {
        auto cells = grid.cells();
        auto particle_id_func = [cells]( size_t cell_idx, size_t p_idx ) -> uint64_t { return cells[cell_idx][field::id][p_idx]; } ;
        extra_data_read_helper.finalize( grid_item , particle_id_func );
      }
    }

    inline void read_optional_data_from_stream( const uint8_t* stream_start , size_t stream_size )
    {
      if( enable_extra_data )
      {
        extra_data_read_helper.read_from_stream( stream_start , stream_size );
      }
    }

    inline void append_cell_particle( size_t cell_idx, size_t p_idx )
    {
      if( enable_extra_data )
      {
        auto cells = grid.cells();
        extra_data_read_helper.append_cell_particle( cell_idx , p_idx , cells[cell_idx][field::id][p_idx] );
      }
    }

    inline size_t optional_cell_data_size(size_t cell_index)
    {
      if( enable_extra_data )
      {
        assert( cell_index < grid_item.m_data.size() );
				constexpr size_t header_size = 2 * sizeof(uint64_t); // WARNING
        return header_size + grid_item.m_data[cell_index].storage_size();
      }
      else { return 0; }
    }

		inline void write_optional_cell_data(uint8_t* buff, const size_t cell_index)
		{
      assert( cell_index < grid_item.m_data.size() );
			auto& cell = grid_item.m_data[cell_index];
			cell.encode_cell_to_buffer((void*)buff);
		}

    template<class WriteFuncT>
    inline size_t write_optional_header( WriteFuncT write_func )
    {
      return 0;
    }
    
    template<class ReadFuncT>
    inline size_t read_optional_header( ReadFuncT read_func )
    {      
      return 0;
    }

    inline StorageType encode( const TupleT & tp )
    {
      update_sats( tp );
      return tp;
    }

    inline TupleT decode( const StorageType & stp )
    {
      update_sats( stp );
      return stp;
    }
  };
}

