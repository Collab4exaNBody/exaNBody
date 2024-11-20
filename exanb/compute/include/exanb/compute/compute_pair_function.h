/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <exanb/compute/compute_pair_buffer.h>
#include <onika/thread.h>
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


