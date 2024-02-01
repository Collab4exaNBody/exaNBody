#pragma once

#include <onika/plot1d.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <regex>
#include <type_traits>
#include <mpi.h>

//#include <exanb/compute/math_functors.h>
// allow field combiner to be processed as standard field
//ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

namespace exanb
{

  namespace SliceParticleFieldTools
  {
    template<class ParticleFieldAccessorT,class FieldT>
    struct SliceAccumulatorFunctor
    {
      const ParticleFieldAccessorT m_pacc;
      const FieldT m_field;
      const Vec3d m_domain_size = { 0.0 , 0.0 , 0.0 };
      const IJK m_repeat = { 0 , 0 , 0 };
      const Mat3d m_xform = { 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 };
      const Vec3d m_direction = { 0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_center = { 0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_u = { 0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_v = { 0.0 , 0.0 , 0.0 };
      const double m_proj_radius2 = 0.0;
      const double m_thickness = 0.0;
      const double m_start = 0.0;
      const long m_resolution = 0;
      onika::cuda::pair<double,double> * __restrict__ m_slice_data = nullptr;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t c, unsigned int p, double rx, double ry, double rz ) const
      {
        using onika::cuda::clamp;
        for(int i = -m_repeat.i ; i <= m_repeat.i ; i++ )
        for(int j = -m_repeat.j ; j <= m_repeat.j ; j++ )
        for(int k = -m_repeat.i ; k <= m_repeat.k ; k++ )
        {
          const Vec3d r = m_xform * Vec3d{ rx + i*m_domain_size.x , ry + j*m_domain_size.y , rz + k*m_domain_size.z };
          const double r_disc_x = dot( r - m_proj_disc_center , m_proj_disc_u );
          const double r_disc_y = dot( r - m_proj_disc_center , m_proj_disc_v );
          const double r_disc_radius2 = r_disc_x*r_disc_x + r_disc_y*r_disc_y;
          if( r_disc_radius2 <= m_proj_radius2 )
          {
            const double pos = dot( r , m_direction );
            const long slice = clamp( static_cast<long>( ( pos - m_start ) / m_thickness ) , 0l , m_resolution-1l );
            const double value = m_pacc[c][m_field][p]; // .get(c,p,m_field);
            ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].first , 1.0 );
            ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].second , value );
          }
        }
      }
    };

    template<class LDBGT, class ParticleFieldAccessorT, class ParallelExecutionFuncT>
    struct SliceParticleField
    {
      const ParticleFieldAccessorT m_pacc;
      const Vec3d m_domain_size = { 0.0 , 0.0 , 0.0 };
      const IJK m_repeat = { 0 , 0 , 0 };
      const Mat3d m_xform = { 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 };
      const Vec3d m_direction = {0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_center = { 0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_u = { 0.0 , 0.0 , 0.0 };
      const Vec3d m_proj_disc_v = { 0.0 , 0.0 , 0.0 };
      const double m_proj_radius2 = 0.0;
      const double m_thickness = 0.0;
      const double m_start = 0.0;
      const long m_resolution = 0;

      MPI_Comm m_comm;
      ParallelExecutionFuncT& parallel_execution_context;
      onika::Plot1DSet & m_plot_set;
      std::function<bool(const std::string&)> m_field_selector;
      std::function<bool(const std::string&)> m_field_average;
      
      LDBGT& ldbg;
      
      template<class GridT, class FidT >
      inline void operator () ( const GridT& grid , const FidT& proj_field )
      {
        using namespace SliceParticleFieldTools;
        using field_type = std::remove_cv_t< std::remove_reference_t< decltype( m_pacc[0][proj_field][0] ) > >;
        static constexpr FieldSet< field::_rx , field::_ry , field::_rz > slice_field_set = {};
        
        static constexpr bool is_arithmetic = std::is_arithmetic_v<field_type>;
        const auto name = proj_field.short_name();

        if constexpr ( is_arithmetic )
        {
          if( m_field_selector(name) )
          {
            ldbg << "Slice field "<<name << " , average="<<m_field_average(name) <<std::endl;
            auto & plot_data = m_plot_set.m_plots[ name ];
            plot_data.assign( m_resolution , { 0.0 , 0.0 } );
            SliceAccumulatorFunctor<ParticleFieldAccessorT,FidT> func = { m_pacc, proj_field, m_domain_size, m_repeat, m_xform, m_direction, m_proj_disc_center, m_proj_disc_u, m_proj_disc_v, m_proj_radius2, m_thickness, m_start, m_resolution, plot_data.data() };
            compute_cell_particles( grid , false , func , slice_field_set , parallel_execution_context() );
            MPI_Allreduce( MPI_IN_PLACE, plot_data.data() , m_resolution * 2 , MPI_DOUBLE , MPI_SUM , m_comm );
            const bool avg = m_field_average(name);
            for(long i=0;i<m_resolution;i++)
            {
              if( plot_data[i].first > 0.0 )
              {
                if( avg ) plot_data[i].second /= plot_data[i].first;
                // ldbg << "slice #"<<i<<" has "<<plot_data[i].first<<" samples"<<std::endl;
              }
              else
              {
                plot_data[i].second = 0.0;
                // ldbg << "slice #"<<i<<" has no sample"<<std::endl;
              }
              plot_data[i].first = m_start + (i+0.5) * m_thickness;
            }
          }
        }
      }
    };

  } // end of SliceParticleFieldTools namespace

  template<class ParticleFieldAccessorT,class FieldT>
  struct ComputeCellParticlesTraits< SliceParticleFieldTools::SliceAccumulatorFunctor<ParticleFieldAccessorT,FieldT> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class LDBGT, class GridT, class ParticleAcessorT, class ParallelExecutionFuncT, class... FieldsOrCombiners>
  static inline void slice_grid_particles(
    LDBGT& ldbg,
    MPI_Comm comm,
    const GridT& grid,
    const Domain& domain,
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
    
    Mat3d xform = domain.xform();
    Vec3d dom_size = domain.bounds_size();
    Vec3d dom_x = xform * Vec3d{dom_size.x,0.0,0.0};
    Vec3d dom_y = xform * Vec3d{0.0,dom_size.y,0.0};
    Vec3d dom_z = xform * Vec3d{0.0,0.0,dom_size.z};
    double proj_disc_radius = std::max( std::max( norm(dom_x) , norm(dom_y) ) , norm(dom_z) );
    
    // projection disc base vectors U and V
    Vec3d proj_disc_u = Vec3d{0.0,0.0,0.0};
    if( std::fabs(dir.x) <= std::fabs(dir.y) && std::fabs(dir.x) <= std::fabs(dir.z) ) proj_disc_u = Vec3d{1.0,0.0,0.0};
    else if( std::fabs(dir.y) <= std::fabs(dir.x) && std::fabs(dir.y) <= std::fabs(dir.z) ) proj_disc_u = Vec3d{0.0,1.0,0.0};
    else proj_disc_u = Vec3d{0.0,0.0,1.0};
    proj_disc_u = cross( dir , proj_disc_u );
    proj_disc_u = proj_disc_u / norm(proj_disc_u);
    Vec3d proj_disc_v = cross( dir , proj_disc_u );
    proj_disc_v = proj_disc_v / norm(proj_disc_v);
    
    IJK repeat = {0,0,0};
    if( domain.periodic_boundary_x() )
    {
      double repeat_radius = sqrt( dot(dom_x,proj_disc_u) * dot(dom_x,proj_disc_u) + dot(dom_x,proj_disc_v) * dot(dom_x,proj_disc_v) );
      if( repeat_radius > 0.0 ) repeat.i = static_cast<ssize_t>( std::ceil( proj_disc_radius / repeat_radius ) );
    }
    if( domain.periodic_boundary_y() )
    {
      double repeat_radius = sqrt( dot(dom_y,proj_disc_u) * dot(dom_y,proj_disc_u) + dot(dom_y,proj_disc_v) * dot(dom_y,proj_disc_v) );
      if( repeat_radius > 0.0 ) repeat.j = static_cast<ssize_t>( std::ceil( proj_disc_radius / repeat_radius ) );
    }
    if( domain.periodic_boundary_z() )
    {
      double repeat_radius = sqrt( dot(dom_z,proj_disc_u) * dot(dom_z,proj_disc_u) + dot(dom_z,proj_disc_v) * dot(dom_z,proj_disc_v) );
      if( repeat_radius > 0.0 ) repeat.j = static_cast<ssize_t>( std::ceil( proj_disc_radius / repeat_radius ) );
    }

    Vec3d center = ( domain.origin() + domain.extent() ) * 0.5;

    
    onika::cuda::pair<double,double> pos_min_max = { std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() };
    for(int i=0;i<8;i++)
    {
      Vec3d c = { (i&1)==0 ? domain.origin().x : domain.extent().x , (i&2)==0 ? domain.origin().y : domain.extent().y , (i&4)==0 ? domain.origin().z : domain.extent().z };
      double pos = dot( xform * c, dir );
      pos_min_max.first = std::min( pos_min_max.first , pos );
      pos_min_max.second = std::max( pos_min_max.second , pos );
    }

    {
      double tmp[2] = { - pos_min_max.first , pos_min_max.second };
      MPI_Allreduce( MPI_IN_PLACE, tmp , 2 , MPI_DOUBLE , MPI_MAX , comm );
      pos_min_max.first = -tmp[0];
      pos_min_max.second = tmp[1];
    }

    long resolution = std::ceil( ( pos_min_max.second - pos_min_max.first ) / thickness );
    ldbg <<"radius="<<proj_disc_radius<<" , U="<<proj_disc_u<<" , V="<<proj_disc_v<<" , repeat="<<repeat<< " , center="<<center<<std::endl;
    ldbg <<"min="<< pos_min_max.first << " , max=" << pos_min_max.second << " , resolution = "<< resolution << std::endl;

    // project particle quantities to cells
    using GridParticleSlicer = SliceParticleField<LDBGT,ParticleAcessorT,ParallelExecutionFuncT>;
    GridParticleSlicer slicer = { pacc, dom_size, repeat, xform, dir, center, proj_disc_u, proj_disc_v, proj_disc_radius*proj_disc_radius, thickness, pos_min_max.first, resolution, comm, parallel_execution_context , plots, field_selector, field_average, ldbg };
    apply_grid_fields( grid, slicer , fc ... );
  }


}
