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
#include <onika/oarray.h>
#include <onika/dac/stencil.h>
#include <unordered_set>
#include <cassert>
#include <cmath>

#include <onika/dag/dot.h>

namespace onika
{

  namespace dac
  {

    /************************ stencil local dependency graph ************************/
    template<size_t Nd>
    std::unordered_set< oarray_t<int,Nd> > stencil_dep_graph( const AbstractStencil & stencil , size_t grainsize )
    {
      assert( Nd == stencil.m_ndims );      
      std::unordered_set< oarray_t<int,Nd> > nbh_deps;
      size_t nb_elements = stencil.nb_cells();
      const auto box_size = stencil.box_size<Nd>();
      
      oarray_t<int,Nd> grain_block;
      for(size_t i=0;i<Nd;i++) grain_block[i] = grainsize;
      //std::cout<<"grain block = "<<format_array(grain_block)<<std::endl;
      size_t block_elements = domain_size( grain_block );      
      
      for(size_t i=0;i<nb_elements;i++)
      {
        const uint64_t ro_mask_i = stencil.ro_mask(i);
        const uint64_t rw_mask_i = stencil.rw_mask(i);
        if( ( ro_mask_i | rw_mask_i ) != 0 )
        for(size_t j=(i+1);j<nb_elements;j++)
        {
          const uint64_t ro_mask_j = stencil.ro_mask(j);
          const uint64_t rw_mask_j = stencil.rw_mask(j);
          const uint64_t dep_mask = ( (ro_mask_i|rw_mask_i) & rw_mask_j ) | ( (ro_mask_j|rw_mask_j) & rw_mask_i );
          if( dep_mask != 0 )
          {
            const auto rpi = index_to_coord( i , box_size );
            const auto rpj = index_to_coord( j , box_size );
            oarray_t<int,Nd> disp;
            bool nulldisp = true;
            for(size_t k=0;k<Nd;k++)
            {
              disp[k] = rpj[k] - rpi[k];
              nulldisp = nulldisp && (disp[k]==0);
            }
            if( ! nulldisp )
            {
              for(size_t bi=0;bi<block_elements;bi++)
              {
                auto b = index_to_coord( bi , grain_block );
                oarray_t<int,Nd> d;
                bool gnulldisp = true;
                for(size_t k=0;k<Nd;k++)
                {
                  d[k] = int( std::floor( ( ( disp[k] + b[k] ) * 1.0 ) / grainsize ) );
                  gnulldisp = gnulldisp && d[k]==0;
                }
                if( ! gnulldisp )
                {
                  nbh_deps.insert( d );
                  for(size_t k=0;k<Nd;k++) d[k] = -d[k];
                  nbh_deps.insert( d );                
                }
              }
            }
          }
        }
      }
      return nbh_deps;
    }

    template std::unordered_set< oarray_t<int,0> > stencil_dep_graph<0>( const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,1> > stencil_dep_graph<1>( const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,2> > stencil_dep_graph<2>( const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,3> > stencil_dep_graph<3>( const AbstractStencil & , size_t );
    /***********************************************************************************/

    /************************ stencil/co-stencil local dependency graph ************************/
    template<size_t Nd>
    std::unordered_set< oarray_t<int,Nd> > stencil_co_dep_graph( const AbstractStencil & stencil, const AbstractStencil & stencil2 , size_t grainsize )
    {
      assert( Nd == stencil.m_ndims );
      std::unordered_set< oarray_t<int,Nd> > nbh_deps;

      size_t nb_elements = stencil.nb_cells();
      size_t nb_elements2 = stencil2.nb_cells();
      const auto box_size = stencil.box_size<Nd>();
      const auto box_low = stencil.low_corner<Nd>();
      const auto box_size2 = stencil2.box_size<Nd>();
      const auto box_low2 = stencil2.low_corner<Nd>();

      oarray_t<int,Nd> grain_block;
      for(size_t i=0;i<Nd;i++) grain_block[i] = grainsize;
      //std::cout<<"grain block = "<<format_array(grain_block)<<std::endl;
      size_t block_elements = domain_size( grain_block );      

      for(size_t i=0;i<nb_elements;i++)
      {
        const uint64_t ro_mask_i = stencil.ro_mask(i);
        const uint64_t rw_mask_i = stencil.rw_mask(i);
        if( ( ro_mask_i | rw_mask_i ) != 0 )
        for(size_t j=0;j<nb_elements2;j++)
        {
          const uint64_t ro_mask_j = stencil2.ro_mask(j);
          const uint64_t rw_mask_j = stencil2.rw_mask(j);
          const uint64_t dep_mask = ( (ro_mask_i|rw_mask_i) & rw_mask_j ) | ( (ro_mask_j|rw_mask_j) & rw_mask_i );
          if( dep_mask != 0 )
          {
            const auto rpi = array_add( box_low , index_to_coord( i , box_size ) );
            const auto rpj = array_add( box_low2 , index_to_coord( j , box_size2 ) );
            oarray_t<int,Nd> disp;
            for(size_t k=0;k<Nd;k++)
            {
              disp[k] = rpj[k] - rpi[k];;
            }
            for(size_t bi=0;bi<block_elements;bi++)
            {
              auto b = index_to_coord( bi , grain_block );
              oarray_t<int,Nd> d;
              for(size_t k=0;k<Nd;k++) d[k] = int( std::floor( ( ( disp[k] + b[k] ) * 1.0 ) / grainsize ) );
              nbh_deps.insert( d );
            }
            //nbh_deps.insert( disp );
          }
        }
      }
      return nbh_deps;
    }

    template std::unordered_set< oarray_t<int,0> > stencil_co_dep_graph<0>( const AbstractStencil & , const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,1> > stencil_co_dep_graph<1>( const AbstractStencil & , const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,2> > stencil_co_dep_graph<2>( const AbstractStencil & , const AbstractStencil & , size_t );
    template std::unordered_set< oarray_t<int,3> > stencil_co_dep_graph<3>( const AbstractStencil & , const AbstractStencil & , size_t );
    /*****************************************************************/
    
  }
}

