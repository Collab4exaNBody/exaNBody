#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <unistd.h>

namespace exanb
{

  struct PrintMessageNode : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , ""   , DocString{"Message string to print"} );
    ADD_SLOT( bool        , endl , INPUT , true , DocString{"When set to true, a new line is inserted after messase"} );

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        Prints a string to the standard output, and optionaly adds a new line.
        )EOF";
    }

    inline void execute () override final
    {
      lout << *mesg ;
      if( *endl ){ lout << std::endl; }
    }

   inline void yaml_initialize(const YAML::Node& node) override final
   {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["mesg"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
   }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "message", make_simple_operator<PrintMessageNode> );
  }

}

