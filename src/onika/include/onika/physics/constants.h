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

namespace onika
{
  
  namespace physics
  {
    static constexpr double atomicMass = 1.66053904020e-27;  ///< Dalton atomic mass unit in kg
    static constexpr double elementaryCharge = 1.6021892e-19;  ///< Elementary charge in Coulomb
    static constexpr double boltzmann = 1.380662e-23;  ///< Boltzmann constant in m2 kg s-2 K-1
    static constexpr double avogadro = 6.02214199e23;  ///< Avogadro constant in mol-1
    static constexpr double calorie_joules = 4.1868;  ///< calorie in joules
    static constexpr double epsilonZero = 8.854187e-12; ///< Vacuum permittivity

    static constexpr double e = 2.7182818284590452;  ///< Euler's number
    static constexpr double phi = 1.618033988749895;  ///< Golden ratio
    static constexpr double pi = 3.1415926535897932;  ///< Pi    
  }
  
}

