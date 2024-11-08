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

namespace onika { namespace scg
{

  enum SlotDirection
  {
    INPUT=1,
    OUTPUT=2,
    INPUT_OUTPUT=3
  };

  static inline constexpr SlotDirection PRIVATE = INPUT_OUTPUT;

  const char* slot_dir_str(SlotDirection d);

} }

// bridge main objects to another namespace if it helps for the transition to standalone Onika
#ifdef ONIKA_SCG_EXPORT_NAMESPACE
namespace ONIKA_SCG_EXPORT_NAMESPACE
{
	static inline constexpr ::onika::scg::SlotDirection INPUT         = ::onika::scg::INPUT;
	static inline constexpr ::onika::scg::SlotDirection OUTPUT        = ::onika::scg::OUTPUT;
	static inline constexpr ::onika::scg::SlotDirection INPUT_OUTPUT  = ::onika::scg::INPUT_OUTPUT;
	static inline constexpr ::onika::scg::SlotDirection PRIVATE       = ::onika::scg::PRIVATE;
	using SlotDirection = ::onika::scg::SlotDirection;
}
#endif

