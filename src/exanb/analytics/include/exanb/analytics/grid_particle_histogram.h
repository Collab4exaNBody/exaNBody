#pragma once

#include <onika/plot1d.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <regex>
#include <cmath>
#include <type_traits>
#include <mpi.h>

//#include <exanb/compute/math_functors.h>
// allow field combiner to be processed as standard field
//ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

namespace exanb
{

  namespace ParticleHistogramTools
  {
    struct ValueMinMaxFunctor
    {
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & value_min_max , double value, reduce_thread_local_t={} ) const
      {
        using onika::cuda::min;
        using onika::cuda::max;
        value_min_max.first = min( value_min_max.first , value );
        value_min_max.second = max( value_min_max.second , value );
      }
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & value_min_max , onika::cuda::pair<double,double> value, reduce_thread_block_t ) const
      {
        ONIKA_CU_ATOMIC_MIN( value_min_max.first , value.first );
        ONIKA_CU_ATOMIC_MAX( value_min_max.second , value.second );
      }
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( onika::cuda::pair<double,double> & value_min_max , onika::cuda::pair<double,double> value, reduce_global_t ) const
      {
        ONIKA_CU_ATOMIC_MIN( value_min_max.first , value.first );
        ONIKA_CU_ATOMIC_MAX( value_min_max.second , value.second );
      }
    };

    struct HistogramAccumulatorFunctor
    {
      const double m_min = 0.0;
      const double m_max = 1.0;
      const long m_resolution = 1024;
      onika::cuda::pair<double,double> * __restrict__ m_histogram_data = nullptr;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( double value ) const
      {
        using onika::cuda::clamp;
        long bin = m_resolution * (value-m_min) / (m_max-m_min);
        bin = clamp( bin , 0l , m_resolution - 1l );
        ONIKA_CU_ATOMIC_ADD( m_histogram_data[bin].second , 1.0 );
      }
    };

    template<class ParallelExecutionFuncT>
    struct HistogramParticleField
    {
      const long m_resolution = 1024;
      MPI_Comm m_comm;
      ParallelExecutionFuncT parallel_execution_context;
      onika::Plot1DSet & m_plot_set;
      std::function<bool(const std::string&)> m_field_selector;

      template<class GridT, class FidT >
      inline void operator () ( const GridT& grid , const FidT& proj_field )
      {
        using namespace ParticleHistogramTools;
        using field_type = typename FidT::value_type;
        static constexpr FieldSet< typename FidT::Id > hist_field_set = {};
        
        if constexpr ( std::is_arithmetic_v<field_type> )
        {
          const auto name = proj_field.short_name();
          if( m_field_selector(name) )
          {
            std::cout << "Histogram field "<<name<<std::endl;
            onika::cuda::pair<double,double> value_min_max = { std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() };
            ValueMinMaxFunctor min_max_func = {};
            reduce_cell_particles( grid , false , min_max_func , value_min_max , hist_field_set , parallel_execution_context() );
            {
              double tmp[2] = { - value_min_max.first , value_min_max.second };
              MPI_Allreduce( MPI_IN_PLACE, tmp , 2 , MPI_DOUBLE , MPI_MAX , m_comm );
              value_min_max.first = -tmp[0];
              value_min_max.second = tmp[1];
            }
            auto & plot_data = m_plot_set.m_plots[ name ];
            plot_data.assign( m_resolution , { 0.0 , 0.0 } );
            HistogramAccumulatorFunctor hist_func = { value_min_max.first, value_min_max.second, m_resolution, plot_data.data() };
            compute_cell_particles( grid , false , hist_func , hist_field_set , parallel_execution_context() );
            MPI_Allreduce( MPI_IN_PLACE, plot_data.data() , m_resolution * 2 , MPI_DOUBLE , MPI_SUM , m_comm );
            const double s = ( value_min_max.second - value_min_max.first ) / m_resolution;
            for(long i=0;i<m_resolution;i++)
            {
              plot_data[i].first = value_min_max.first + (i+0.5) * s;
            }
          }
        }
      }
    };

  } // end of ParticleHistogramTools namespace

  template<>
  struct ReduceCellParticlesTraits<ParticleHistogramTools::ValueMinMaxFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<>
  struct ComputeCellParticlesTraits< ParticleHistogramTools::HistogramAccumulatorFunctor >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT, class ParallelExecutionFuncT, class... FieldsOrCombiners>
  static inline void grid_particles_histogram(
    MPI_Comm comm,
    const GridT& grid,
    long resolution,
    onika::Plot1DSet& plots,
    std::function<bool(const std::string&)> field_selector,
    ParallelExecutionFuncT parallel_execution_context,
    const FieldsOrCombiners& ... fc )
  {
    using namespace ParticleHistogramTools;

    std::cout << "resolution = "<< resolution << std::endl;
    
    // project particle quantities to cells
    using Histogrammer = HistogramParticleField<ParallelExecutionFuncT>;
    Histogrammer func = { resolution, comm, parallel_execution_context , plots, field_selector };
    apply_grid_fields( grid, func , fc ... );
  }


}
