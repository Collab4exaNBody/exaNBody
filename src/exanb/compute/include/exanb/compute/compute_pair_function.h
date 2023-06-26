#pragma once

#include <exanb/compute/compute_pair_buffer.h>
#include <exanb/core/thread.h>
#include <exanb/field_sets.h>
#include <onika/soatl/field_id.h>
 

namespace exanb
{

  // === Asymetric operator ===
  // all particle neighbors are assembled and passed to operator at once
  
  template<typename FS> class ComputePairOperator;

  template<typename... field_ids>
  class ComputePairOperator< FieldSet<field_ids...> >
  {
    public:
      virtual void operator() ( ComputePairBuffer2<false,false>& tab, typename onika::soatl::FieldId<field_ids>::value_type & ... ) const noexcept =0;
      virtual void operator() ( ComputePairBuffer2<true,false>& tab, typename onika::soatl::FieldId<field_ids>::value_type & ... ) const noexcept =0;
      virtual ~ComputePairOperator() = default;
  };
  
}


