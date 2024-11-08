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
#include <exanb/core/operator.h>
#include <exanb/core/domain.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <onika/math/basic_types_stream.h>
#include <onika/log.h>
#include <exanb/core/print_particle.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <mpi.h>

namespace exanb
{

  template<typename GridT>
  struct ExtendDomainOperator : public OperatorNode
  { 
    using ParticleT = typename GridT::CellParticles::TupleValueType;
    using ParticleVector = std::vector<ParticleT>;

    ADD_SLOT( MPI_Comm       , mpi           , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( ParticleVector , otb_particles , INPUT , ParticleVector{} );
//    ADD_SLOT( bool           , enable_domain_extension, INPUT, false ); // replaced by domain->expandable() property
    ADD_SLOT( GridT          , grid          , INPUT_OUTPUT ); // we'll modify the grid offset in the new domain
    ADD_SLOT( Domain         , domain        , INPUT_OUTPUT );
    ADD_SLOT( bool           , domain_extended, INPUT_OUTPUT );

    inline void execute () override final
    {
      if( ! domain->expandable() )
      {
        *domain_extended = false;
        return;
      }

      if( domain->periodic_boundary_x() && domain->periodic_boundary_y() && domain->periodic_boundary_z() )
      {
        if( domain->expandable() )
        {
          lerr << "Domain is fully periodic, thus it cannot be expanded. Set expandable to false" << std::endl;
          domain->set_expandable( false );
        }
        *domain_extended = false;
        return;
      }

      size_t n = otb_particles->size();
      const ParticleT * __restrict__ particles = otb_particles->data();

      const IJK dims = domain->grid_dimension();
      const double cell_size = domain->cell_size();
      const double cell_size_sqr = cell_size * cell_size;

      ldbg << "nb otb particles = " << n<< std::endl;
      ldbg << "bounds = " << domain->bounds() << std::endl;
      double xmin = domain->bounds().bmax.x ;
      double ymin = domain->bounds().bmax.y ;
      double zmin = domain->bounds().bmax.z ;
      double xmax = domain->bounds().bmin.x ;
      double ymax = domain->bounds().bmin.y ;
      double zmax = domain->bounds().bmin.z ;

      // first: apply domain periodic to otb particles
#     pragma omp parallel for reduction(min:xmin,ymin,zmin) reduction(max:xmax,ymax,zmax)
      for(size_t i=0; i<n; ++i )
      {
        const Vec3d r { particles[i][field::rx] , particles[i][field::ry] , particles[i][field::rz] };
        //domain_periodic_location( *domain, r );
        xmin = std::min( xmin , r.x );
        ymin = std::min( ymin , r.y );
        zmin = std::min( zmin , r.z );
        xmax = std::max( xmax , r.x );
        ymax = std::max( ymax , r.y );
        zmax = std::max( zmax , r.z );

        const double domain_dist2 = min_distance2_between( r , domain->bounds() );
        if( domain_dist2 > cell_size_sqr )
        {
          print_particle( lerr , particles[i] , false );
          fatal_error()
            << "********************************************\n"
            << "Suspicious out of domain particle\n"
            << "bounds         = "<<domain->bounds()<<"\n"
            << "dist to domain = "<<sqrt(domain_dist2)<<"\n"
            << "cell size      = "<<cell_size<<"\n"
            << "********************************************"<<std::endl;
        }

      }

      ldbg << "grid dims = " << grid->dimension()<< " , domain dims = "<<dims<< std::endl;
      ldbg << "min = " << xmin<<','<<ymin<<','<<zmin << std::endl;
      ldbg << "max = " << xmax<<','<<ymax<<','<<zmax << std::endl;
      ldbg << "cell_size = " << cell_size << std::endl;

      // compute domain cell dims and offset from particle bounds
      IJK grid_min = {0,0,0} , grid_max = {0,0,0};      
      if( n > 0 )
      {
        grid_min = IJK{
          std::min( 0l , static_cast<ssize_t>( std::floor( ( xmin - domain->bounds().bmin.x ) / cell_size ) ) ),
          std::min( 0l , static_cast<ssize_t>( std::floor( ( ymin - domain->bounds().bmin.y ) / cell_size ) ) ),
          std::min( 0l , static_cast<ssize_t>( std::floor( ( zmin - domain->bounds().bmin.z ) / cell_size ) ) )
          };

        grid_max = IJK{
          std::max( dims.i , static_cast<ssize_t>( std::ceil( ( xmax - domain->bounds().bmin.x ) / cell_size ) ) ),
          std::max( dims.j , static_cast<ssize_t>( std::ceil( ( ymax - domain->bounds().bmin.y ) / cell_size ) ) ),
          std::max( dims.k , static_cast<ssize_t>( std::ceil( ( zmax - domain->bounds().bmin.z ) / cell_size ) ) )
          };
      }
      else
      {
        grid_min = IJK{ 0, 0, 0 };
        grid_max = dims;
      }

      ldbg << "grid_min (local) : " << grid_min << std::endl;
      ldbg << "grid_max (local) : " << grid_max << std::endl;
      
      if( std::min(std::min(grid_min.i,grid_min.j),grid_min.k) < -1 || std::max(std::max(grid_max.i-dims.i,grid_max.j-dims.j),grid_max.k-dims.k) > 1 )
      {
        fatal_error()
          << "********************************************\n"
          << "Suspicious domain extension, more than one cell increase : min="<<grid_min<<" , max="<<grid_max<<"\n"
          << "********************************************"<<std::endl;
      }
      
      size_t max_n_otb = n;
      {
        long tmp[7] = { -grid_min.i , -grid_min.j , -grid_min.k , grid_max.i , grid_max.j , grid_max.k , static_cast<long>(max_n_otb) };
        MPI_Allreduce(MPI_IN_PLACE,tmp,7,MPI_LONG,MPI_MAX,*mpi);
        grid_min.i = -tmp[0];
        grid_min.j = -tmp[1];
        grid_min.k = -tmp[2];
        grid_max.i = tmp[3];
        grid_max.j = tmp[4];
        grid_max.k = tmp[5];
        max_n_otb  = tmp[6];
      }
      ldbg << "max_n_otb      : " << max_n_otb << std::endl;
      ldbg << "grid_min (all) : " << grid_min << std::endl;
      ldbg << "grid_max (all) : " << grid_max << std::endl;
      if( max_n_otb == 0 )
      {
        // nothing to do
        return ;
      }
      
      if( domain->periodic_boundary_x() )
      {
        grid_min.i = 0;
        grid_max.i = dims.i;
      }

      if( domain->periodic_boundary_y() )
      {
        grid_min.j = 0;
        grid_max.j = dims.j;
      }

      if( domain->periodic_boundary_z() )
      {
        grid_min.k = 0;
        grid_max.k = dims.k;
      }
      ldbg << "grid_min (periodic) : " << grid_min << std::endl;
      ldbg << "grid_max (periodic) : " << grid_max << std::endl;

      IJK grid_size = grid_max - grid_min;
      Vec3d origin = domain->bounds().bmin;

      ldbg << "grid shift : " << grid_min << std::endl;
      ldbg << "grid size  : " << dims << " -> " << grid_size << std::endl;      
      ldbg << "origin : "<<origin<<" -> "<<(origin + grid_min*cell_size)<<std::endl;
      ldbg << "offset : "<<grid->offset()<<" -> "<< (grid->offset() - grid_min) << std::endl;

      origin = origin + grid_min*cell_size;
      grid->set_offset( grid->offset() - grid_min );
      grid->set_origin( origin );
      domain->set_grid_dimension( grid_size );
      domain->set_bounds( { origin , origin + grid_size * cell_size } );

      *domain_extended = ( grid_size.i>dims.i || grid_size.j>dims.j || grid_size.k>dims.k );
    }
  };
  
   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory("extend_domain", make_grid_variant_operator< ExtendDomainOperator> );
  }

}

