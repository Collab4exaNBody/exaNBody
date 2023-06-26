#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>

namespace exanb
{

  template<class ValueType>
  struct PrintValue : public OperatorNode
  {  
    ADD_SLOT( std::string , prefix , INPUT , "" , DocString{"text to print before"} );
    ADD_SLOT( ValueType , value , INPUT , REQUIRED , DocString{"input value to print"} );
    ADD_SLOT( std::string , suffix , INPUT , "" , DocString{"text to print after"} );

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        Prints a value to the standard output
        )EOF";
    }

    inline void execute () override final
    {
      lout << *prefix << *value << *suffix;
    }

   inline void yaml_initialize(const YAML::Node& node) override final
   {
      YAML::Node tmp;
      if( node.IsScalar() )
      {
        tmp["value"] = node;
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
   }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "print_real", make_simple_operator< PrintValue<double> > );
    OperatorNodeFactory::instance()->register_factory( "print_int", make_simple_operator< PrintValue<long> > );
    OperatorNodeFactory::instance()->register_factory( "print_bool", make_simple_operator< PrintValue<bool> > );
  }

}

