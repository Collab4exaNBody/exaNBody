#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/analytics/grid_particle_histogram.h>
#include <exanb/fields.h>

#include <mpi.h>

namespace exanb
{

  template<class GridT>
  class GridParticleHistogram : public OperatorNode
  {      
    using StringList = std::vector<std::string>;
    using Plot1DSet = onika::Plot1DSet;
    using StringMap = std::map<std::string,std::string>;

    ADD_SLOT( MPI_Comm   , mpi       , INPUT , REQUIRED );
    ADD_SLOT( GridT      , grid      , INPUT , REQUIRED );
    ADD_SLOT( long       , samples   , INPUT , 1024 );
    ADD_SLOT( StringList , fields    , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to slice"} );
    ADD_SLOT( StringMap  , caption   , INPUT , StringMap{} , DocString{"map plot names to optional captions"} );
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
      const auto& flist = *fields;
      auto field_selector = [&flist] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;
      auto pecfunc = [op=this]() { return op->parallel_execution_context(); };
      grid_particles_histogram( *mpi, *grid,  *samples, *plots, field_selector, pecfunc, grid_fields... );
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
      return R"EOF(create 1D Plots from histograms of particle field values)EOF";
    }    

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(grid_particle_histogram)
  {
   OperatorNodeFactory::instance()->register_factory( "grid_particle_histogram" , make_grid_variant_operator< GridParticleHistogram > );
  }

}

