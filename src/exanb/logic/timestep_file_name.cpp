#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>

#include <memory>

namespace exanb
{

  class TimeStepFileNameOperator : public OperatorNode
  {  
    ADD_SLOT( long        , timestep , INPUT , REQUIRED );
    ADD_SLOT( std::string , format   , INPUT , REQUIRED );
    ADD_SLOT( std::string , filename , OUTPUT );

  public:
    inline void execute() override final
    {
      *filename = format_string( *format , *timestep );
      ldbg << "timestep file = " << *filename <<std::endl;
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["format"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };
 
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "timestep_file", make_compatible_operator< TimeStepFileNameOperator > );
  }

}

