#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/grid_cell_particles/grid_particle_field_accessor.h>

#include <regex>
#include <mpi.h>

//#include <exanb/compute/math_functors.h>
// allow field combiner to be processed as standard field
//ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

namespace exanb
{
  
  struct Plot1DSet
  {
    std::map< std::string , std::vector< std::pair<double,double> > > m_plots;
  };

  struct DirectionalMinMaxDistanceFunctor
  {
    const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
    const Vec3d m_direction = {1.0 , 0.0 , 0.0 };
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( std::pair<double,double> & pos_min_max , double rx, double ry, double rz , reduce_thread_local_t={} ) const
    {
      double pos = dot( m_xform * Vec3d{rx,ry,rz} , m_direction );
      pos_min_max.first = min( pos_min_max.first , pos );
      pos_min_max.second = max( pos_min_max.second , pos );
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( std::pair<double,double> & pos_min_max , std::pair<double,double> value, reduce_thread_block_t ) const
    {
      ONIKA_CU_ATOMIC_MIN( pos_min_max.first , value.first );
      ONIKA_CU_ATOMIC_MAX( pos_min_max.second , value.second );
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( std::pair<double,double> & pos_min_max , std::pair<double,double> value, reduce_global_t ) const
    {
      ONIKA_CU_ATOMIC_MIN( pos_min_max.first , value.first );
      ONIKA_CU_ATOMIC_MAX( pos_min_max.second , value.second );
    }
  };

  template<> struct ReduceCellParticlesTraits<exanb::DirectionalMinMaxDistanceFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };

  namespace SliceParticleFieldTools
  {
    template<class T, bool = false > struct FieldTypeScalarWeighting : std::false_type {};
    template<class T> struct FieldTypeScalarWeighting<T, std::is_same_v<typeof(T{}*1.0),double> > : std::true_type {}; 
    template<class T> static inline constexpr bool type_scalar_weighting_v = FieldTypeScalarWeighting<T>::value ;

    template<ParticleFieldAccessorT,FieldT>
    struct SliceAccumulatorFunctor
    {
      const ParticleFieldAccessorT m_pacc;
      const FieldT m_field;
      const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
      const Vec3d m_direction = { 1.0 , 0.0 , 0.0 };
      const double pos_min = 0.0;
      const double pos_max = 0.0;
      const long m_resolution = 1024;
      std::pair<double,double> * __restrict__ m_slice_data = nullptr;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t c, unsigned int p, double rx, double ry, double rz ) const
      {
        const double pos = dot( m_xform * Vec3d{rx,ry,rz} , m_direction );
        const long slice = clamp( static_cast<long>( m_resolution * ( pos - pos_min ) / ( pos_max - pos_min ) ) , 0l , m_resolution-1l );
        const double value = m_pacc.get(c,p,m_field);
        ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].first , 1.0 );
        ONIKA_CU_ATOMIC_ADD( m_slice_data[slice].second , value );
      }
    };
  }

  template<class ParticleFieldAccessorT>
  struct SliceParticleField
  {
    const ParticleFieldAccessorT m_pacc;
    const Mat3d m_xform = { 1.0 , 0.0 , 0.0 , 0.0 , 1.0 , 0.0 , 0.0 , 0.0 , 1.0 };
    const Vec3d m_direction = {1.0 , 0.0 , 0.0 };
    const double m_thickness = 1.0;
    const double m_start = 0.0;
    const double m_end = 0.0;
    const long m_resolution = 1024;

    MPI_Comm m_comm;
    OperatorNode* m_op = nullptr;
    Plot1DSet& m_plot_set;
    std::function<bool(const std::string&)> m_field_selector;
    
    template<class GridT, class FidT >
    inline void operator () ( GridT& grid , const FidT& proj_field )
    {
      using namespace SliceParticleFieldTools;
      using field_type = decltype( pacc.get(0,0,proj_field) );
      static constexpr FieldSet< field::_rx , field::_ry , field::_rz > slice_field_set = {};
      
      if constexpr ( type_scalar_weighting_v<field_type> )
      {
        const auto name = proj_field.short_name();
        if( m_field_selector(name) )
        {
          ldbg << "Slice field "<<name<<std::endl;
          m_slice_count.assign( m_resolution , 0 );
          auto & plot_data = m_plot_set.m_plots[ name ];
          plot_data.assign( m_resolution , { 0.0 , 0.0 } );
          SliceAccumulatorFunctor<ParticleFieldAccessorT,FidT> func = { m_pacc, proj_field, m_xform,  m_direction, m_start, m_end, m_resolution, plot_data.data() };
          compute_cell_particles( grid , false , func , slice_field_set , m_op->parallel_execution_context() );
          MPI_Allreduce( MPI_IN_PLACE, plot_data.data() , m_resolution * 2 , MPI_DOUBLE , MPI_SUM , m_comm );
          for(long i=0;i<m_resolution;i++)
          {
            if( plot_data[i].first > 0.0 ) plot_data[i].second /= plot_data[i].first;
            else plot_data[i].second = 0.0;
            plot_data[i].first = m_start + (i+0.5) * ( m_end - m_start ) / m_resolution;
          }
        }
      }
    }
  };


  template< class GridT >
  class GridParticleSlicing : public OperatorNode
  {
    using StringList = std::vector<std::string>;
    static constexpr FieldSet<field::_rx,field::_ry,field::_rz> reduce_field_set {};
    
    ADD_SLOT( MPI_Comm   , mpi        , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT      , grid       , INPUT , REQUIRED );
    ADD_SLOT( Domain     , domain     , INPUT , REQUIRED );
    ADD_SLOT( double     , thickness  , INPUT , 1.0 );
    ADD_SLOT( Vec3d      , direction  , INPUT , Vec3d{1,0,0} );
    ADD_SLOT( long       , resolution , INPUT , 1 );
    ADD_SLOT( StringList , fields     , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to project"} );
    ADD_SLOT( Plot1DSet  , plots      , INPUT_OUTPUT );

  public:

    // -----------------------------------------------
    inline void execute ()  override final
    {
      using namespace ParticleCellProjectionTools;
      if( grid->number_of_cells() == 0 ) return;

      const auto& flist = *fields;
      auto field_selector = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;

      Vec3d dir = normalize( *direction );
      std::pair<double,double> pos_min_max = { std::numeric_limits<double>::max() , std::numeric_limits<double>::lowest() };
      DirectionalMinMaxDistanceFunctor func = { domain->xform() , dir };
      reduce_cell_particles( *grid , false , func , pos_min_max , reduce_field_set , parallel_execution_context() );
      {
        double tmp[2] = { - pos_min_max.first , pos_min_max.second };
        MPI_Allreduce( MPI_IN_PLACE, tmp , 2 , MPI_DOUBLE , MPI_MAX , *mpi );
        pos_min_max.first = -tmp[0];
        pos_min_max.second = tmp[1];
      }

      ldbg << "min="<< pos_min_max.first << " , max=" << pos_min_max.second << std::endl;
      
      // project particle quantities to cells
      using ParticleAcessor = GridParticleFieldAccessor<typename GridT::CellParticles *>;
      SliceParticleField<ParticleAcessor> slice_fields = { {grid->cells()} , domain->xform() , dir , *thickness , pos_min_max.first, pos_min_max.second , *resolution , *plots , field_selector };
      apply_grid_field_set( *grid, slice_fields , GridT::field_set );
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(project particle quantities onto a set of regularly spaced slices)EOF";
    }    

  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory("grid_particle_slicing", make_grid_variant_operator< GridParticleSlicing > );
  }

}
