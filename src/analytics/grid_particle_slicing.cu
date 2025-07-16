#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <onika/math/basic_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <onika/plot1d.h>
#include <exanb/analytics/grid_particle_slicing.h>

#include <regex>
#include <type_traits>
#include <mpi.h>

//#include <exanb/compute/math_functors.h>
// allow field combiner to be processed as standard field
//ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

namespace exanb
{

  template< class GridT >
  class GridParticleSlicing : public OperatorNode
  {
    using StringList = std::vector<std::string>;
    using Plot1DSet = onika::Plot1DSet;
    using StringMap = std::map<std::string,std::string>;
    static constexpr FieldSet<field::_rx,field::_ry,field::_rz> reduce_field_set {};
    
    ADD_SLOT( MPI_Comm   , mpi       , INPUT , MPI_COMM_WORLD );
    ADD_SLOT( GridT      , grid      , INPUT , REQUIRED );
    ADD_SLOT( Domain     , domain    , INPUT , REQUIRED );
    ADD_SLOT( double     , thickness , INPUT , 1.0 );
    ADD_SLOT( Vec3d      , direction , INPUT , Vec3d{1,0,0} );
    ADD_SLOT( StringList , fields    , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to slice"} );
    ADD_SLOT( StringMap  , caption   , INPUT , StringMap{} , DocString{"map field names to output plot names"} );
    ADD_SLOT( StringList , average   , INPUT , StringMap{} , DocString{"for each fields, indicate if normalization is needed (divide by number of cnotributions)"} );
    ADD_SLOT( Plot1DSet  , plots     , INPUT_OUTPUT );

  public:

    inline void execute ()  override final
    {    
      execute_on_field_set(grid->field_set);
    }
    
  private:
    template<class... GridFields>
    inline void execute_on_fields( const GridFields& ... grid_fields) 
    {
      ldbg << "grid_particle_slicing: fields =";
      for(const auto& f: *fields) ldbg << " "<< f;
      ldbg << std::endl;
      
      const auto& flist = *fields;
      const auto& avg = *average;
      auto field_selector = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; };
      auto field_average = [&avg] ( const std::string& name ) -> bool { for(const auto& f:avg) if( std::regex_match(name,std::regex(f)) ) return true; return false; };
      auto pecfunc = [op=this]() { return op->parallel_execution_context(); };
      //using ParticleAcessor = GridParticleFieldAccessor<typename GridT::CellParticles *>;
      /*ParticleAcessor*/ auto pacc = grid->cells_accessor(); //{grid->cells()};
      slice_grid_particles( *mpi, *grid, *domain, *direction, *thickness, *plots, pacc, field_selector, field_average, pecfunc, grid_fields... );
      for(const auto& f: *fields) { plots->m_captions[ f ] = f; }
      for(const auto& c: *caption) { plots->m_captions[ c.first ] = c.second; }
    }

    template<class... fid>
    inline void execute_on_field_set( FieldSet<fid...> ) 
    {
      execute_on_fields( onika::soatl::FieldId<fid>{} ... );
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(create 1D Plots from particles fields, averaging those fields by slices of given orientation and thickness)EOF";
    }    

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(grid_particle_slicing)
  {
    OperatorNodeFactory::instance()->register_factory("grid_particle_slicing", make_grid_variant_operator< GridParticleSlicing > );
  }

}
