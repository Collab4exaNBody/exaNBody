#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/domain.h>

namespace exanb
{

  class BackupXForm : public OperatorNode
  {
    ADD_SLOT( Domain , domain       , INPUT , REQUIRED );  // 
    ADD_SLOT( Mat3d  , backup_xform , INPUT_OUTPUT);

  public:
    inline void execute ()  override final
    {    
      *backup_xform = domain->xform();
    }

  };

 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "backup_xform", make_simple_operator< BackupXForm > );
  }

}

