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

#include <onika/cuda/cuda.h>
#include <onika/math/basic_types_def.h>
#include <exanb/compute/math_functors.h>
#include <onika/soatl/field_combiner.h>

// definition of a virtual field, a.k.a a field combiner
ONIKA_DECLARE_FIELD_COMBINER( exanb, ParticleCountCombiner , count , exanb::ConstantFunctor<exanb::ConstReal1> )
ONIKA_DECLARE_FIELD_COMBINER( exanb, ProcessorRankCombiner , processor_id , exanb::UniformValueFunctor<int> )

ONIKA_DECLARE_FIELD_COMBINER( exanb, PositionVec3Combiner  , position , exanb::Vec3FromXYZFunctor , exanb::field::_rx , exanb::field::_ry , exanb::field::_rz )
ONIKA_DECLARE_FIELD_COMBINER( exanb, VelocityVec3Combiner  , velocity , exanb::Vec3FromXYZFunctor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )
ONIKA_DECLARE_FIELD_COMBINER( exanb, ForceVec3Combiner     , force    , exanb::Vec3FromXYZFunctor , exanb::field::_fx , exanb::field::_fy , exanb::field::_fz )

