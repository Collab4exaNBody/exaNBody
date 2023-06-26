#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

#include <fstream>

namespace exanb
{

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class FileExists : public OperatorNode
  {
  public:
  
    ADD_SLOT( std::string , filename , INPUT , REQUIRED );
    ADD_SLOT( bool , result , INPUT_OUTPUT );
    
    void execute() override final
    {
	std::ifstream fin(*filename);
	*result = fin.good();
    }
  };

   // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "file_exists", make_compatible_operator< FileExists > );
  }

}

