#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/grid_cell_particles/particle_region.h>

namespace exanb
{

  class InitParticleRegions : public OperatorNode
  {  
    ADD_SLOT( ParticleRegions  , particle_regions , INPUT_OUTPUT );
    ADD_SLOT( bool , verbose , INPUT , false );

  public:
    inline bool is_sink() const override final { return true; }
  
    inline void execute() override final
    {
      if( *verbose )
      {
        lout << "======= Particle regions ========"<< std::endl;
        for(const auto& r : (*particle_regions) )
        {
          lout << r.name() << std::endl ;
          if( r.m_id_range_flag ) lout<< "\tid range : [ " << r.m_id_start <<" ; "<<r.m_id_end << " [" << std::endl;
          //else lout << "\tid range : <none>" << std::endl;
          
          if( r.m_bounds_flag ) lout << "\tbounds   : " << r.m_bounds << std::endl;
          //else lout << "\tbounds   : <none>" << std::endl;
          
          if( r.m_quadric_flag ) lout << "\tquadric  : " << r.m_quadric << std::endl;
          //else lout << "\tquadric  : <none>" << std::endl;
        }
        lout << "================================="<< std::endl << std::endl;   
      }
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsSequence() )
      {
        tmp["particle_regions"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }


  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "particle_regions", make_compatible_operator<InitParticleRegions> );
  }

}

