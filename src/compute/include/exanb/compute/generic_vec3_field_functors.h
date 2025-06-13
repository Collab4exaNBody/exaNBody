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
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/compute/compute_cell_particles.h>

// definition of a virtual field, a.k.a a field combiner
namespace exanb
{
  
  template<class OP>
  struct GenericVec3Functor
  {
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double & x, double & y, double & z ) const
    {
      Vec3d VEC = {x,y,z};
      op( VEC );
      x = VEC.x;
      y = VEC.y;
      z = VEC.z;
    }
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericVec3Functor<OP> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class OP>
  struct GenericVec3RegionFunctor
  {
    const ParticleRegionCSGShallowCopy region;
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double rx, double ry, double rz, const uint64_t id, double & x, double & y, double & z ) const
    {
      if( region.contains( onika::math::Vec3d{rx,ry,rz} , id ) )
      {
        Vec3d VEC = {x,y,z};
        op( VEC );
        x = VEC.x;
        y = VEC.y;
        z = VEC.z;
      }
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double rx, double ry, double rz, double & x, double & y, double & z ) const
    {
      this->operator () (rx,ry,rz,0,x,y,z);
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( const uint64_t id, double & x, double & y, double & z ) const
    {
      this->operator () (x,y,z,id,x,y,z);
    }    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double & x, double & y, double & z ) const
    {
      this->operator () (x,y,z,0,x,y,z);
    }    
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericVec3RegionFunctor<OP> >
  {
    static inline constexpr bool CudaCompatible = true;
  };


  template<class OP>
  struct GenericIndirectVec3Functor
  {
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t t, double & x, double & y, double & z ) const
    {
      Vec3d VEC = {x,y,z};
      op( t , VEC );
      x = VEC.x;
      y = VEC.y;
      z = VEC.z;
    }
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericIndirectVec3Functor<OP> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class OP>
  struct GenericIndirectVec3RegionFunctor
  {
    const ParticleRegionCSGShallowCopy region;
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t t, double rx, double ry, double rz, const uint64_t id, double & x, double & y, double & z ) const
    {
      if( region.contains( onika::math::Vec3d{rx,ry,rz} , id ) )
      {
        Vec3d VEC = {x,y,z};
        op( t , VEC );
        x = VEC.x;
        y = VEC.y;
        z = VEC.z;
      }
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t t, double rx, double ry, double rz, double & x, double & y, double & z ) const
    {
      this->operator () (t,rx,ry,rz,0,x,y,z);
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t t, const uint64_t id, double & x, double & y, double & z ) const
    {
      this->operator () (t,x,y,z,id,x,y,z);
    }    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t t, double & x, double & y, double & z ) const
    {
      this->operator () (t,x,y,z,0,x,y,z);
    }    
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericIndirectVec3RegionFunctor<OP> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

}


