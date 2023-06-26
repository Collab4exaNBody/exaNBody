#pragma once

#include <onika/cuda/cuda.h>
#include <exanb/core/basic_types_def.h>
#include <exanb/compute/math_functors.h>
#include <onika/soatl/field_combiner.h>

// definition of a virtual field, a.k.a a field combiner
ONIKA_DECLARE_FIELD_COMBINER( exanb, ParticleCountCombiner , count , exanb::ConstantFunctor<exanb::ConstReal1> )
ONIKA_DECLARE_FIELD_COMBINER( exanb, ProcessorRankCombiner , processor_id , exanb::UniformValueFunctor<int> )

