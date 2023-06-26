#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct InputInteger : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , "input: " );
    ADD_SLOT( bool , endl , INPUT , false );
    ADD_SLOT( long , value , OUTPUT , 0 );

    inline void execute () override final
    {
      lout << *mesg << " [" <<*value<<"] ";
      if( *endl ) lout << std::endl;
      lout << std::flush;

      std::string qstr;
      std::getline( std::cin , qstr );
      if( ! qstr.empty() )
      {
        *value = std::stol( qstr );
      }
      ldbg << "integer = '"<< *value << "'" << std::endl;
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "input_integer", make_simple_operator<InputInteger> );
  }

}

