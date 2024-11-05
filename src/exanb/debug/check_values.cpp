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
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/log.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/parallel_random.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/file_utils.h>
#include <onika/yaml/yaml_utils.h>
#include <exanb/core/domain.h>
#include <exanb/core/quantity.h>
#include <exanb/core/unityConverterHelper.h>

#include <onika/soatl/field_tuple.h>

#include <exanb/core/yaml_check_particles.h>

#include <mpi.h>

#include <set>
#include <string>
//#include <sstream>
//#include <limits>
#include <cstdlib>
#include <ctime>

namespace exanb
{
  
  
  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_id, field::_ax, field::_ay, field::_az >
    >
  struct CheckValuesNode : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( Domain      , domain    , INPUT , REQUIRED );
    ADD_SLOT( GridT       , grid      , INPUT , REQUIRED );
    ADD_SLOT( std::string , file      , INPUT , REQUIRED );
    ADD_SLOT( long        , samples   , INPUT , 128 );
    ADD_SLOT( double      , acc_threshold , INPUT );
    ADD_SLOT( double      , pos_threshold , INPUT );
    ADD_SLOT( double      , vel_threshold , INPUT );
    ADD_SLOT( double      , threshold , INPUT , 1.e-16 );
    ADD_SLOT( bool        , v1_compatible , INPUT , false );
    ADD_SLOT( bool        , fatal , INPUT , true );

    inline void execute () override final
    {
      GridT& grid = *(this->grid);
      MPI_Comm comm = *mpi;

      const Mat3d mat = domain->xform();
      const Mat3d inv_mat = domain->inv_xform();

      double acc_max_err = *threshold;
      double pos_max_err = *threshold;
      double vel_max_err = 1.0e40;
      if( acc_threshold.has_value() ) { acc_max_err = *acc_threshold; }
      if( pos_threshold.has_value() ) { pos_max_err = *pos_threshold; }
      if( vel_threshold.has_value() ) { vel_max_err = *vel_threshold; }

      int nprocs = 1;
      int rank = 0;
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&nprocs);

      // read reference file
      std::set<ParticleReferenceValue> reference_values_set;
      std::set<uint64_t> selected_ids;
      std::vector<ParticleReferenceValue> reference_values;
      
      std::string file_name = data_file_path( *file );
      
      bool create = false;
      {
        std::ifstream fin( file_name );
        if( ! fin ) { create = true; }
      }

      if( create )
      {
        lout << "\n\n**********************\n**********************\n";
        lout << "** Can't open reference file "<<file_name<<", create a new one\n";
        lout << "**********************\n**********************\n"<<std::endl;
      }
      else
      {
        lout << "\n\n**********************\n**********************\n";
        lout << "** Reference file "<<file_name<<"\n";
        YAML::Node raw_data = yaml_load_file_abort_on_except(file_name);
        double length_scale = 1.0;

        
        YAML::Node data;
        if( raw_data.IsMap() ) { data = raw_data; }
        else
        {
          lout << "** Warning: reference file uses obsolete format. This format will be unsupported in future releases."<<std::endl;
          data["values"] = raw_data;
          data["length_unit"] = "1.0 nm"; 
        }
        if( data["length_unit"] ) { length_scale = data["length_unit"].as<Quantity>().convert(); }
        reference_values = data["values"].as< std::vector<ParticleReferenceValue> >();

        lout << "** length scale = "    << length_scale            << "\n";
        lout << "** num. ref. values = "<< reference_values.size() <<std::endl;
        for( auto& v : reference_values )
        {
          v.m_r[0] *= length_scale;
          v.m_r[1] *= length_scale;
          v.m_r[2] *= length_scale;
          v.m_a[0] *= length_scale;
          v.m_a[1] *= length_scale;
          v.m_a[2] *= length_scale;
        }
        reference_values_set.insert( reference_values.begin(), reference_values.end() );
      }

      auto cells = grid.cells();
      IJK dims = grid.dimension();
      ssize_t gl = grid.ghost_layers();
      long number_of_samples = std::min( ( (*samples) + nprocs-1) / nprocs , (long) (grid.number_of_particles() - grid.number_of_ghost_particles() ) );
      long total_number_of_samples = -1;      
#     ifndef NDEBUG
      size_t n_cells = grid.number_of_cells();      
#     endif
      if( create )
      {
        std::uniform_int_distribution<> rand_i( gl, dims.i - gl - 1 );
        std::uniform_int_distribution<> rand_j( gl, dims.j - gl - 1 );
        std::uniform_int_distribution<> rand_k( gl, dims.k - gl - 1 );

        auto& re = rand::random_engine();

        std::vector<long> local_ids;
        for(long i=0;i<number_of_samples;)
        {
          IJK loc { rand_i(re), rand_j(re), rand_k(re) };
          assert( loc.i>=0 && loc.i<dims.i );
          assert( loc.j>=0 && loc.j<dims.j );
          assert( loc.k>=0 && loc.k<dims.k );
          size_t cell_index = grid_ijk_to_index( dims , loc );
          assert( cell_index>=0 && cell_index<n_cells );
          size_t n_particles = cells[cell_index].size();
          if( n_particles >= 1 )
          {
            std::uniform_int_distribution<> rand_p(0,n_particles-1);
            size_t j = rand_p( rand::random_engine() );
            uint64_t id = cells[cell_index][field::id][j];
            if( selected_ids.find(id) == selected_ids.end() )
            {
              local_ids.push_back( id );
              selected_ids.insert( id );
              ++ i;
              //std::cout << "\tCheckValuesNode: " << i << "/" << number_of_samples << " échantillons trouvés" << std::endl;
            }
          }
        }
        selected_ids.clear();
        ldbg << "number_of_samples=" << number_of_samples << ", local_ids.size()="<<local_ids.size()<< std::endl;
        assert( number_of_samples == static_cast<long>( local_ids.size() ) );

        MPI_Allreduce( MPI_IN_PLACE , &number_of_samples , 1 , MPI_LONG , MPI_MIN , comm);
        assert( number_of_samples >= 1 );
        assert( number_of_samples <= static_cast<long>( local_ids.size() ) );
        local_ids.resize( number_of_samples );
        total_number_of_samples = number_of_samples * nprocs;
        ldbg << "number_of_samples=" << number_of_samples << std::endl;

        assert( static_cast<long>( local_ids.size() ) == number_of_samples );
        std::vector<long> all_ids(total_number_of_samples);
        MPI_Allgather( local_ids.data() , number_of_samples , MPI_LONG , all_ids.data() , number_of_samples , MPI_LONG , comm );
        selected_ids.clear();
        selected_ids.insert( all_ids.begin() , all_ids.end() );
        if( all_ids.size() != selected_ids.size() )
        {
          std::unordered_map<uint64_t,size_t> check_ids;
          size_t all_ids_size = all_ids.size();
          for(size_t i=0;i<all_ids_size;i++)
          {
            if( check_ids.find(all_ids[i]) != check_ids.end() )
            {
              lerr << "duplicate id #"<< all_ids[i]<< " at position "<<i<<", P="<<(i/number_of_samples) 
                   << ", previous id from P"<<check_ids[all_ids[i]]/number_of_samples << std::endl;
            }
            check_ids[all_ids[i]] = i;
          }
        }
        if( total_number_of_samples != static_cast<long>(selected_ids.size()) )
        {
          lerr << "Warning: total_number_of_samples=" << total_number_of_samples << " different from selected_ids.size()="<<selected_ids.size() <<std::endl;
        }
        total_number_of_samples = selected_ids.size();
      }

      ldbg << "bounds="<<domain->bounds() <<std::endl;

      double r_l2_norm = 0.0;
      double a_l2_norm = 0.0;
      double v_l2_norm = 0.0;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN( dims-2*gl, _, loc, reduction(+:r_l2_norm,a_l2_norm,v_l2_norm) )
        {
          size_t i = grid_ijk_to_index( dims , loc+gl );
          const uint64_t* __restrict__ part_ids = cells[i][field::id];

          const double* __restrict__ rx = cells[i][field::rx];
          const double* __restrict__ ry = cells[i][field::ry];
          const double* __restrict__ rz = cells[i][field::rz];

          const double* __restrict__ ax = cells[i][field::ax];
          const double* __restrict__ ay = cells[i][field::ay];
          const double* __restrict__ az = cells[i][field::az];

          const double* __restrict__ vx = cells[i][field::vx];
          const double* __restrict__ vy = cells[i][field::vy];
          const double* __restrict__ vz = cells[i][field::vz];

          size_t n_part = cells[i].size();
          for(size_t j=0;j<n_part;j++)
          {
            Vec3d r { rx[j], ry[j], rz[j] };            
            domain_periodic_location( *domain , r ); // test must be indepedant from move_particles frequency
            Vec3d phys_r = mat * r;

            if( ! create )
            {
              auto it = reference_values_set.find( ParticleReferenceValue{part_ids[j]} );
              if( it != reference_values_set.end() )
              {
                // Vec3d ref_r = Vec3d{ it->m_r[0], it->m_r[1], it->m_r[2] };
                //domain_periodic_location( *domain , ref_r );

                Vec3d ref_phys_r = Vec3d{ it->m_r[0], it->m_r[1], it->m_r[2] };
                Vec3d ref_r = inv_mat * ref_phys_r;
                ref_r = find_periodic_closest_point( ref_r , r , domain->bounds() );
                double r_err2 = norm2( r - ref_r );
                
                double dax = ax[j] - it->m_a[0];
                double day = ay[j] - it->m_a[1];
                double daz = az[j] - it->m_a[2];
                double a_err2 = dax*dax + day*day + daz*daz;

                double dvx = vx[j] - it->m_v[0];
                double dvy = vy[j] - it->m_v[1];
                double dvz = vz[j] - it->m_v[2];
                double v_err2 = dvx*dvx + dvy*dvy + dvz*dvz;

                r_l2_norm += r_err2;
                a_l2_norm += a_err2;
                v_l2_norm += v_err2;
                
                if( std::sqrt(r_err2) > pos_max_err || std::sqrt(a_err2) > acc_max_err || std::sqrt(v_err2) > vel_max_err )
                {
#                 pragma omp critical(debug_error_particle)
                  {
                    if( *fatal )
                    {
                      lerr << "particle #"<<part_ids[j]<<" : re="<<std::sqrt(r_err2)<<"/"<<pos_max_err<<" ae="<<std::sqrt(a_err2)<<"/"<< acc_max_err<<" ve="<<std::sqrt(v_err2)<<"/"<< vel_max_err<<std::endl
                           << " : a="<<Vec3d{ax[j],ay[j],az[j]}<<" aref="<<Vec3d{it->m_a[0],it->m_a[1],it->m_a[2]}<<std::endl
                           << " : v="<<Vec3d{vx[j],vy[j],vz[j]}<<" vref="<<Vec3d{it->m_v[0],it->m_v[1],it->m_v[2]}<<std::endl
                           << " : r="<<r<<" rref="<<ref_r<<std::endl;
                      std::abort();
                    }
                    else
                    {
                    ldbg << "particle #"<<part_ids[j]<<" : re="<<std::sqrt(r_err2)<<"/"<<pos_max_err<<" ae="<<std::sqrt(a_err2)<<"/"<< acc_max_err<<std::endl
                         << " : a="<<Vec3d{ax[j],ay[j],az[j]}<<" aref="<<Vec3d{it->m_a[0],it->m_a[1],it->m_a[2]}<<std::endl
                         << " : v="<<Vec3d{vx[j],vy[j],vz[j]}<<" vref="<<Vec3d{it->m_v[0],it->m_v[1],it->m_v[2]}<<std::endl
                         << " : r="<<r<<" rref="<<ref_r<<std::endl;
                    }
                  }
                }
              }
            }
            else if( selected_ids.find(part_ids[j]) != selected_ids.end() )
            {
#             pragma omp critical(add_reference_value)
              {
                // std::cout << "part #"<<part_ids[j]<<" : a="<<ax[j]<<','<<ay[j]<<','<<az[j]<<std::endl;
                ParticleReferenceValue value { part_ids[j] , {phys_r.x,phys_r.y,phys_r.z} , {ax[j],ay[j],az[j]} , {vx[j],vy[j],vz[j]} };
                reference_values.push_back( value );
              }
            }
          }
        }
        GRID_OMP_FOR_END
      }

      // sum errors across all processors
      if( nprocs > 1 )
      {
        double tmp[3] = { r_l2_norm, a_l2_norm , v_l2_norm };
        MPI_Allreduce( MPI_IN_PLACE , tmp , 3 , MPI_DOUBLE , MPI_SUM , comm);
        r_l2_norm = tmp[0];
        a_l2_norm = tmp[1];
        v_l2_norm = tmp[2];
      }

      if( create )
      {
        if( nprocs > 1 )
        {
          int data_size = reference_values.size();
          int all_data_sizes[nprocs];
          MPI_Allgather( &data_size, 1 , MPI_INT , all_data_sizes , 1 , MPI_INT , comm );
          assert( all_data_sizes[rank] == data_size );
          
          ldbg << "data_size=" << data_size << std::endl;
          
          int displs[nprocs];
          long tot = 0;
          for(int i=0;i<nprocs;i++)
          {
            displs[i] = tot * sizeof(ParticleReferenceValue);
            tot += all_data_sizes[i];
            all_data_sizes[i] *= sizeof(ParticleReferenceValue);
          }

          ldbg << "tot=" << tot << std::endl;
          std::vector<ParticleReferenceValue> all_values(tot);
          
          MPI_Gatherv( (char*) reference_values.data() , data_size*sizeof(ParticleReferenceValue) , MPI_CHAR , (char*) all_values.data() , all_data_sizes , displs , MPI_CHAR , 0, comm );
          if( rank == 0 )
          {
            reference_values = all_values;
          }
        }
        if( rank == 0 )
        {
          ldbg << "reference_values.size()=" << reference_values.size() << std::endl;
          assert( reference_values.size() == static_cast<size_t>(total_number_of_samples) );

          // get current date and time to store it in generated files
          time_t rawtime;
          struct tm * timeinfo;
          char buffer[256];
          time (&rawtime);
          timeinfo = localtime(&rawtime);
          strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
          std::string date_str(buffer);

          std::sort( reference_values.begin(), reference_values.end() );
          std::ofstream fout(file_name);
          const double r_scale = 1.0;
          const double a_scale = 1.0;
          const double v_scale = 1.0;

          fout << "date: '" << date_str << "'" << std::endl;
          fout << "length_unit: 1.0 "<< exanb::units::internal_unit_system.length().short_name() << std::endl;
          fout << "values:" << std::endl;

          for(auto p:reference_values)
          {
            p.m_r[0] *= r_scale;
            p.m_r[1] *= r_scale;
            p.m_r[2] *= r_scale;
            p.m_a[0] *= a_scale;
            p.m_a[1] *= a_scale;
            p.m_a[2] *= a_scale;
            p.m_v[0] *= v_scale;
            p.m_v[1] *= v_scale;
            p.m_v[2] *= v_scale;
            fout << "- " << p << std::endl;
          }
          fout.close();
        }
      }
      else
      {
        a_l2_norm = std::sqrt(a_l2_norm);
        r_l2_norm = std::sqrt(r_l2_norm);
        v_l2_norm = std::sqrt(v_l2_norm);
        lout << "** Position error L2 norm = "<<r_l2_norm << " / " << pos_max_err <<std::endl;
        lout << "** Acceleration error L2 norm = "<<a_l2_norm << " / " << acc_max_err <<std::endl;
        lout << "** Velocity error L2 norm = "<<v_l2_norm<< " / " << vel_max_err <<std::endl;
        lout << "*********************\n"<<std::endl;
        if( ( *fatal ) && ( a_l2_norm > acc_max_err || r_l2_norm > pos_max_err || v_l2_norm > vel_max_err) )
        {
          std::abort();
        }
      }

    }
  };
 
  template<class GridT> using CheckValuesNodeTmpl = CheckValuesNode<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "check_values", make_grid_variant_operator< CheckValuesNodeTmpl > );
  }

}

