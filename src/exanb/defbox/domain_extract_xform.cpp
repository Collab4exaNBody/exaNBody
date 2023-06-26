#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/domain.h>

namespace exanb
{

  struct DomainExtractXForm : public OperatorNode
  {
    ADD_SLOT( Domain     , domain  , INPUT , REQUIRED );
    ADD_SLOT( Mat3d      , xform   , OUTPUT ); // outputs deformation matrix ( not combined with domain's matrix)
    ADD_SLOT( Mat3d      , inv_xform , OUTPUT ); // outputs inverse of deformation matrix

    inline void execute () override final
    {   
      *xform = domain->xform();
      *inv_xform = domain->inv_xform();
    }
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "domain_extract_xform", make_compatible_operator< DomainExtractXForm > );
  }

}


