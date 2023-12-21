#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/units.h>

#include <iostream>

namespace exanb
{

  struct InputValueNode : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , "input: " );
    ADD_SLOT( bool , endl , INPUT , false );
    ADD_SLOT( double , value , OUTPUT , 0.0 );

    inline void execute () override final
    {
      lout << *mesg << " ["<< *value << "] ";
      if( *endl ) lout << std::endl;
      lout << std::flush;
      
      std::string qstr;
      std::getline( std::cin , qstr );
      if( ! qstr.empty() )
      {
        *value = exanb::units::quantity_from_string( qstr ).convert();
      }
      ldbg << "value = '"<< *value << "'" << std::endl;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "input_value", make_simple_operator<InputValueNode> );
  }

}

