#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/quaternion_operators.h>
#include <onika/cuda/cuda.h>

namespace exanb
{

  namespace ParticleCellProjectionTools
  {
    template<class T, class = void> struct FieldTypeSupportWeighting : std::false_type {};
    template<class T> struct FieldTypeSupportWeighting<T, decltype(void(sizeof(T{}*1.0))) > : std::true_type {}; 
    template<class T> static inline constexpr bool type_support_weighting_v = FieldTypeSupportWeighting<T>::value ;
  
    using StringList = std::vector<std::string>;
  
    ONIKA_HOST_DEVICE_FUNC static inline void localize_subcell( const Vec3d& r, double cell_size, double sub_cellsize, ssize_t subdiv, IJK& cell_loc, IJK& subcell_loc )
    {
      cell_loc = make_ijk( r / cell_size );
      Vec3d ro = r - (cell_loc*cell_size);
      subcell_loc = vclamp( make_ijk(ro / sub_cellsize) , 0 , subdiv-1 );
    }

    static inline void subcell_neighbor( const IJK& cell_loc, const IJK& subcell_loc, ssize_t subdiv, IJK ninc, IJK& nbh_cell_loc, IJK& nbh_subcell_loc )
    {
      nbh_cell_loc = cell_loc;
      nbh_subcell_loc = subcell_loc + ninc;
      if(nbh_subcell_loc.i<0) { -- nbh_cell_loc.i; } else if(nbh_subcell_loc.i>=subdiv) { ++ nbh_cell_loc.i; }
      if(nbh_subcell_loc.j<0) { -- nbh_cell_loc.j; } else if(nbh_subcell_loc.j>=subdiv) { ++ nbh_cell_loc.j; }
      if(nbh_subcell_loc.k<0) { -- nbh_cell_loc.k; } else if(nbh_subcell_loc.k>=subdiv) { ++ nbh_cell_loc.k; }
      nbh_subcell_loc.i = ( nbh_subcell_loc.i + subdiv ) % subdiv;
      nbh_subcell_loc.j = ( nbh_subcell_loc.j + subdiv ) % subdiv;
      nbh_subcell_loc.k = ( nbh_subcell_loc.k + subdiv ) % subdiv;      
    }

    // @return how much of this particle contributes to region cell_box.
    // sum of contributions for all disjoint cell_box paving the domain is guaranteed to be 1.0
    static inline double particle_weight(const Vec3d& r, double sp_size, const AABB& cell_box)
    {
      AABB contrib_box = { r - sp_size*0.5 , r + sp_size*0.5 };
      AABB sub_contrib_box = intersection( contrib_box , cell_box );
      double w = 0.0;
      if( ! is_nil(sub_contrib_box) ) { w = bounds_volume(sub_contrib_box) / (sp_size*sp_size*sp_size); }
      assert( w>=0. && w<=(1.0+1.e-11) );
      return w;
    }

    struct CreateCellValueField
    {
      GridCellValues& m_cell_values;
      std::function<bool(const std::string&)> m_field_selector;
      const ssize_t m_subdiv = 1;

      template<class GridT, class FidT>
      inline void operator () ( GridT& grid , const FidT& proj_field)
      {
        using field_type = typename FidT::value_type;
        if constexpr ( type_support_weighting_v<field_type> )
        {
          unsigned int ncomp = 1;
          if constexpr ( std::is_same_v<field_type,Vec3d> ) ncomp=3;
          if constexpr ( std::is_same_v<field_type,IJK>   ) ncomp=3;
          if constexpr ( std::is_same_v<field_type,Mat3d> ) ncomp=9;
          const auto name = proj_field.short_name();
          if( m_field_selector(name) )
          {
            if( ! m_cell_values.has_field(name) )
            {
              ldbg << "Add field "<<name<<std::endl;
              m_cell_values.add_field(name, m_subdiv , ncomp );
            }
          }
        }
        else
        {
          ldbg << "Field " << proj_field.short_name() << " ignored (not projectable)" << std::endl;
        }
      }
    };

    struct CollectCellValueFieldToAdd
    {
      GridCellValues& m_cell_values;
      std::vector< AddCellFieldInfo >& m_fields_to_add;
      std::function<bool(const std::string&)> m_field_selector;
      const ssize_t m_subdiv = 1;

      template<class GridT, class FidT>
      inline void operator () ( GridT& grid , const FidT& proj_field)
      {
        using field_type = typename FidT::value_type;
        if constexpr ( type_support_weighting_v<field_type> )
        {
          unsigned int ncomp = 1;
          if constexpr ( std::is_same_v<field_type,Vec3d> ) ncomp=3;
          if constexpr ( std::is_same_v<field_type,IJK>   ) ncomp=3;
          if constexpr ( std::is_same_v<field_type,Mat3d> ) ncomp=9;
          const auto name = proj_field.short_name();
          if( m_field_selector(name) )
          {
            if( ! m_cell_values.has_field(name) )
            {
              ldbg << "Add field "<<name<<std::endl;
              //m_cell_values.add_field(name, m_subdiv , ncomp );
              m_fields_to_add.push_back( { name, static_cast<size_t>(m_subdiv), ncomp } );
            }
          }
        }
        else
        {
          ldbg << "Field " << proj_field.short_name() << " ignored (not projectable)" << std::endl;
        }
      }
    };


    template<class ParticleFieldAccessor>
    struct ProjectCellValueField
    {
      const ParticleFieldAccessor pacc;
      GridCellValues& m_cell_values;
      std::function<bool(const std::string&)> m_field_selector;
      const double m_splat_size = 1.0;
      const ssize_t m_subdiv = 1;
      
      template<class GridT, class FidT >
      inline void operator () ( GridT& grid , const FidT& proj_field )
      {
        using field_type = decltype( pacc.get(0,0,proj_field) );
        if constexpr ( type_support_weighting_v<field_type> )
        {
          using const_ncomp = std::integral_constant<unsigned int
                , ( std::is_same_v<field_type,Vec3d> || std::is_same_v<field_type,IJK> ) ? 3
                : ( std::is_same_v<field_type,Mat3d> ? 9
                : ( std::is_same_v<field_type,Quaternion> ? 4
                : ( std::is_arithmetic_v<field_type> ? 1 : 0 ) ) ) >;
          static constexpr unsigned int ncomp = const_ncomp::value;
          
          auto cells = grid.cells();
          const IJK dims = grid.dimension();
          //const ssize_t gl = grid.ghost_layers();      
          const double cell_size = grid.cell_size();
          const double subcell_size = cell_size / m_subdiv;
  #       ifndef NDEBUG
          const ssize_t n_cells = grid.number_of_cells();
          const ssize_t n_subcells = m_subdiv * m_subdiv * m_subdiv;
  #       endif

          const auto name = proj_field.short_name();

          if( m_field_selector(name) )
          {
            ldbg << "Project field "<<name<<std::endl;
            double * __restrict__ ptr = nullptr;
            assert( size_t(m_subdiv) == m_cell_values.field(name).m_subdiv );
            assert( size_t(m_subdiv * m_subdiv * m_subdiv * ncomp) == m_cell_values.field(name).m_components );
            auto accessor = m_cell_values.field_data(name);
            ptr = accessor.m_data_ptr;
            const size_t stride = accessor.m_stride;

            if( ncomp == 0 )
            {
              lerr << "Warning: Selected field "<<name<<" cannot be projected to cell values, it will be set to 0" << std::endl;
            }
            
            // computes per cell Mass, per cell Mass*Velocity^2, per cell x-component of Momentum and per cell x-component or Virial
  #         pragma omp parallel
            {
  #           pragma omp for schedule(static)
              for(ssize_t i = 0; i < dims.i*dims.j*dims.k; ++i)
              {
                for(ssize_t j = 0; j < m_subdiv*m_subdiv*m_subdiv; ++j)
                {
		              size_t index = i * stride + j * ncomp;
	                for(unsigned int k=0;k<ncomp;k++) ptr[ index + k ] = 0.0;
                }
              }

              GRID_OMP_FOR_BEGIN(dims,i,cell_loc, schedule(dynamic) )
              {
                const Vec3d cell_origin = grid.cell_position( cell_loc );
                const auto* __restrict__ rx = cells[i][field::rx];
                const auto* __restrict__ ry = cells[i][field::ry];
                const auto* __restrict__ rz = cells[i][field::rz];
                const unsigned int n = cells[i].size();
                for(unsigned int j=0;j<n;j++)
                {
                  Vec3d r = { rx[j] , ry[j] , rz[j] };
                  IJK center_cell_loc;
                  IJK center_subcell_loc;
                  Vec3d rco = r - cell_origin;
                  localize_subcell( rco, cell_size, subcell_size, m_subdiv, center_cell_loc, center_subcell_loc );
                  center_cell_loc += cell_loc;

                  for(int ck=-1;ck<=1;ck++)
                  for(int cj=-1;cj<=1;cj++)
                  for(int ci=-1;ci<=1;ci++)
                  {
                    IJK nbh_cell_loc;
                    IJK nbh_subcell_loc;
                    subcell_neighbor( center_cell_loc, center_subcell_loc, m_subdiv, IJK{ci,cj,ck}, nbh_cell_loc, nbh_subcell_loc );
                    if( grid.contains(nbh_cell_loc) )
                    {
                      ssize_t nbh_cell_i = grid_ijk_to_index( dims , nbh_cell_loc );
                      ssize_t nbh_subcell_i = grid_ijk_to_index( IJK{m_subdiv,m_subdiv,m_subdiv} , nbh_subcell_loc );
                      assert( nbh_cell_i>=0 && nbh_cell_i<n_cells );
                      assert( nbh_subcell_i>=0 && nbh_subcell_i<n_subcells );

                      // compute weighted contribution of particle to sub cell
                      Vec3d nbh_cell_origin = grid.cell_position(nbh_cell_loc);
                      AABB subcell_box = { nbh_cell_origin + nbh_subcell_loc*subcell_size , nbh_cell_origin + (nbh_subcell_loc+1)*subcell_size };
                      const double w = particle_weight(r, m_splat_size, subcell_box);
                      using FieldContribType = decltype( field_type{} * w );
                      [[maybe_unused]] FieldContribType field_contrib;
                      if constexpr ( ncomp > 0 )
                      {
                        field_contrib = pacc.get(i,j,proj_field); // cells[i][proj_field][j] * w;
                      }
                      if constexpr ( ncomp == 1 )
                      {
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i ] += field_contrib;
                      }
                      if constexpr ( ncomp == 3 )
                      {
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 3 + 0 ] += field_contrib.x;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 3 + 1 ] += field_contrib.y;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 3 + 2 ] += field_contrib.z;
                      }
                      if constexpr ( ncomp == 4 )
                      {
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 4 + 0 ] += field_contrib.w;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 4 + 1 ] += field_contrib.x;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 4 + 2 ] += field_contrib.y;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 4 + 3 ] += field_contrib.z;
                      }
                      if constexpr ( ncomp == 9 )
                      {
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 0 ] += field_contrib.m11;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 1 ] += field_contrib.m12;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 2 ] += field_contrib.m13;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 3 ] += field_contrib.m21;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 4 ] += field_contrib.m22;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 5 ] += field_contrib.m23;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 6 ] += field_contrib.m31;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 7 ] += field_contrib.m32;
  #                     pragma omp atomic update
                        ptr[ nbh_cell_i * stride + nbh_subcell_i * 9 + 8 ] += field_contrib.m33;
                      }
                    }
                  }
                }
              }
              GRID_OMP_FOR_END;
            }
          }
        }// if type support weighting        
      }
    };

  } // 
  
} // ecanb

