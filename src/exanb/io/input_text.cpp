#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>

#include <iostream>

namespace exanb
{

  struct InputTextNode : public OperatorNode
  {  
    ADD_SLOT( std::string , mesg , INPUT , "input: " );
    ADD_SLOT( bool , endl , INPUT , false );
    ADD_SLOT( std::string , text , OUTPUT , "" );

    inline void execute () override final
    {
      lout << *mesg ;
      if( ! text->empty() ) lout << " ["<<*text<<"] ";
      if( *endl ) lout << std::endl;
      lout << std::flush;
      
      std::string s;
      std::getline( std::cin , s );
      if( !s.empty() ) *text = s;
      ldbg << "text = '"<< *text << "'" << std::endl;
    }

  };
    
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "input_text", make_simple_operator<InputTextNode> );
  }

}

