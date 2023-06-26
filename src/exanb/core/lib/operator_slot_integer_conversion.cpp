#include <exanb/core/operator_slot_base.h>
#include <exanb/core/cpp_utils.h>
#include <exanb/core/log.h>
#include <cstdint>
#include <cstring>

namespace exanb
{
    
  // === register factories ===  
  CONSTRUCTOR_ATTRIB void _exanb_register_integer_conversions()
  {
    ldbg << "register signed/unsigned integer conversions" << std::endl;
    OperatorSlotBase::register_type_conversion_force_cast<int64_t,uint64_t>();
    OperatorSlotBase::register_type_conversion_force_cast<uint64_t,int64_t>();
    OperatorSlotBase::register_type_conversion_force_cast<int32_t,uint32_t>();
    OperatorSlotBase::register_type_conversion_force_cast<uint32_t,int32_t>();

    int64_t x = -( (1ll<<30) + 1 );
    int64_t* px64 = &x;
    int32_t* px32 = nullptr;
    assert( sizeof(px32) == sizeof(px64) );
    std::memcpy( &px32 , &px64 , sizeof(px64) ); // according to c++ norm, this is ok while a reinterpret_cast is not
    bool endianness_ok = ( *px32 == *px64 );

    if( endianness_ok )
    {
      ldbg << "register 64 bits to 32 bits integer conversions" << std::endl;
      OperatorSlotBase::register_type_conversion_force_cast<int64_t ,int32_t >();
      OperatorSlotBase::register_type_conversion_force_cast<int64_t ,uint32_t>();
      OperatorSlotBase::register_type_conversion_force_cast<uint64_t,int32_t >();
      OperatorSlotBase::register_type_conversion_force_cast<uint64_t,uint32_t>();
    }
    else
    {
      lerr << "Warning: no 64 bits to 32 bits integer conversion" << std::endl;
    }
  }
  
}

