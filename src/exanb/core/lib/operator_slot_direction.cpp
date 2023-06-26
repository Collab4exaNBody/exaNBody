
#include <exanb/core/operator_slot_direction.h>

namespace exanb
{

  const char* slot_dir_str(SlotDirection d)
  {
    switch(d)
    {
      case INPUT: return "IN";
      case OUTPUT: return "OUT";
      case INPUT_OUTPUT: return "IN/OUT";
    }
    return "<error>";
  }

}

