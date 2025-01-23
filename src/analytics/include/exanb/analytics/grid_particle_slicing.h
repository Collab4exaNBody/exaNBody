#pragma once

#include <onika/plot1d.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <regex>
#include <type_traits>
#include <mpi.h>

namespace exanb
{

  namespace SliceParticleFieldTools
  {
    struct DirectionalMinMaxDistanceFunctor
    {
      const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
      const Vec3d m_direction = {1.0 , 0.0 , 0.0 };
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & pos_min_max , double rx, double ry, double rz , reduce_thread_local_t={} ) const
      {
        using onika::cuda::min;
        using onika::cuda::max;
        double pos = dot( m_xform * Vec3d{rx,ry,rz} , m_direction );
        pos_min_max.first = min( pos_min_max.first , pos );
        pos_min_max.second = max( pos_min_max.second , pos );
      }
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & pos_min_max , onika::cuda::pair<double,double> value, reduce_thread_block_t ) const
      {
        ONIKA_CU_ATOMIC_MIN( pos_min_max.first , value.first );
        ONIKA_CU_ATOMIC_MAX( pos_min_max.second , value.second );
      }
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & pos_min_max , onika::cuda::pair<double,double> value, reduce_global_t ) const
      {
        ONIKA_CU_ATOMIC_MIN( pos_min_max.first , value.first );
        ONIKA_CU_ATOMIC_MAX( pos_min_max.second , value.second );
      }
    };

    template<class ParticleFieldAccessorT,class FieldT>
    struct SliceAccumulatorFunctor
    {
      const ParticleFieldAccessorT m_pacc;
      const FieldT m_field;
      const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
      const Vec3d m_direction = { 1.0 , 0.0 , 0.0 };
      const double m_thickness = 1.0;
      const double m_start = 0.0;
      const long m_resolution = 1;
      onika::cuda::pair<double,double> * __restrict__ m_slice_data = nullptr;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t c, unsigned int p, double rx, double ry, double rz ) const
      {
        using onika::cuda::clamp;
        const double pos = dot( m_xform * Vec3d{rx,ry,rz} , m_direction );
        const long slice = clamp( static_cast<long>( ( pos - m_start ) / m_thickness ) , 0l , m_resolution-1l );
        const double value = m_pacc[c][m_field][p]; // .get(c,p,m_field);
        ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].first , 1.0 );
        ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].second , value );
      }
    };

    template<class ParticleFieldAccessorT, class ParallelExecutionFuncT>
    struct SliceParticleField
    {
      const ParticleFieldAccessorT m_pacc;
      const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
      const Vec3d m_direction = {1.0 , 0.0 , 0.0 };
      const double m_thickness = 1.0;
      const double m_start = 0.0;
      const long m_resolution = 1024;

      MPI_Comm m_comm;
      ParallelExecutionFuncT& parallel_execution_context;
      onika::Plot1DSet & m_plot_set;
      std::function<bool(const std::string&)> m_field_selector;
      std::function<bool(const std::string&)> m_field_average;
      
      template<class GridT, class FidT >
      inline void operator () ( const GridT& grid , const FidT& proj_field )
      {
        using namespace SliceParticleFieldTools;
        using field_type = decltype( m_pacc[0][proj_field][0] /*m_pacc.get(0,0,proj_field)*/ );
        
        if constexpr ( std::is_arithmetic_v<field_type> )
        {
          static constexpr FieldSet< field::_rx , field::_ry , field::_rz > slice_field_set = {};
          const auto name = proj_field.short_name();
          if( m_field_selector(name) )
          {
            std::cout << "Slice field "<<name<<std::endl;
            auto & plot_data = m_plot_set.m_plots[ name ];
            plot_data.assign( m_resolution , { 0.0 , 0.0 } );
            SliceAccumulatorFunctor<ParticleFieldAccessorT,FidT> func = { m_pacc, proj_field, m_xform,  m_direction, m_thickness, m_start, m_resolution, plot_data.data() };
            compute_cell_particles( grid , false , func , slice_field_set , parallel_execution_context() );
            MPI_Allreduce( MPI_IN_PLACE, plot_data.data() , m_resolution * 2 , MPI_DOUBLE , MPI_SUM , m_comm );
            const bool avg = m_field_average(name);
            for(long i=0;i<m_resolution;i++)
            {
              if( plot_data[i].first > 0.0 )
              {
                if( avg ) plot_data[i].second /= plot_data[i].first;
              }
              else
              {
                plot_data[i].second = 0.0;
              }
              plot_data[i].first = m_start + (i+0.5) * m_thickness;
            }
          }
        }
      }
    };

  } // end of SliceParticleFieldTools namespace

  template<>
  struct ReduceCellParticlesTraits<SliceParticleFieldTools::DirectionalMinMaxDistanceFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
#   ifdef ONIKA_HAS_GPU_ATOMIC_MIN_MAX_DOUBLE
    static inline constexpr bool CudaCompatible = true;
#   else
    static inline constexpr bool CudaCompatible = false;
#   endif
  };

  template<class ParticleFieldAccessorT,class FieldT>
  struct ComputeCellParticlesTraits< SliceParticleFieldTools::SliceAccumulatorFunctor<ParticleFieldAccessorT,FieldT> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT, class ParticleAcessorT, class ParallelExecutionFuncT, class... FieldsOrCombiners>
  static inline void slice_grid_particles(
    MPI_Comm comm,
    const GridT& grid,
    const Mat3d& xform,
    const Vec3d& direction,
    double thickness,
    onika::Plot1DSet& plots,
    ParticleAcessorT pacc,
    std::function<bool(const std::string&)> field_selector,
    std::function<bool(const std::string&)> field_average,
    ParallelExecutionFuncT parallel_execution_context,
    const FieldsOrCombiners& ... fc )
  {
    using namespace SliceParticleFieldTools;
  
    const Vec3d dir = direction / norm(direction);
    onika::cuda::pair<double,double> pos_min_max = { std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() };
    DirectionalMinMaxDistanceFunctor func = { xform , dir };
    reduce_cell_particles( grid , false , func , pos_min_max , FieldSet<field::_rx,field::_ry,field::_rz>{} , parallel_execution_context() );
    {
      double tmp[2] = { - pos_min_max.first , pos_min_max.second };
      MPI_Allreduce( MPI_IN_PLACE, tmp , 2 , MPI_DOUBLE , MPI_MAX , comm );
      pos_min_max.first = -tmp[0];
      pos_min_max.second = tmp[1];
    }

    long resolution = std::ceil( ( pos_min_max.second - pos_min_max.first ) / thickness );
    std::cout << "min="<< pos_min_max.first << " , max=" << pos_min_max.second << " , resolution = "<< resolution << std::endl;
    
    // project particle quantities to cells
    using GridParticleSlicer = SliceParticleField<ParticleAcessorT,ParallelExecutionFuncT>;
    GridParticleSlicer slicer = { pacc, xform, dir, thickness, pos_min_max.first, resolution, comm, parallel_execution_context , plots, field_selector, field_average };
    apply_grid_fields( grid, slicer , fc ... );
  }


}
