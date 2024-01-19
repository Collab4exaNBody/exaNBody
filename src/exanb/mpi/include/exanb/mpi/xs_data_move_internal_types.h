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

#include <exanb/mpi/xs_data_move_types.h>

namespace XsDataMove
{

  struct IdLocalization
  {
      id_type m_id;
      index_type m_index;
      int m_owner;
  };

  struct IdMove
  {
      index_type m_src_index;
      index_type m_dst_index;
      int m_src_owner;
      int m_dst_owner;
  };

  /*
    Helper class to serialize Id provenance and destination info in a single buffer
  */
  class IdLocalizationSerializeBuffer
  {
    public:
      static inline size_type bufferSize( size_type nBefore, size_type nAfter )
      {
          return sizeof(IdLocalizationSerializeBuffer) + (nBefore+nAfter)*sizeof(IdLocalization);
      }

      // WARNING: DO NOT delete objects allocated this way if allocPtr was not nullptr
      static inline IdLocalizationSerializeBuffer* alloc( size_type nBefore, size_type nAfter, char* allocPtr = nullptr )
      {
          if( allocPtr == nullptr )
          {
              allocPtr = new char[ bufferSize(nBefore,nAfter) ];
          }
          return new( allocPtr ) IdLocalizationSerializeBuffer(nBefore,nAfter);
      }

      // WARNING: DO NOT delete objects bound to a memory location
      static inline IdLocalizationSerializeBuffer* bindTo( char* ptr )
      {
          return new( ptr ) IdLocalizationSerializeBuffer();
      }

      inline size_type beforeCount() const { return m_ids_before_count; }
      inline size_type afterCount() const { return m_ids_after_count; }

      inline IdLocalization& beforeId(index_type i) { return m_ids[i]; }
      inline IdLocalization& afterId(index_type i) { return m_ids[i+beforeCount()]; }
      
      inline size_type size() const { return bufferSize( beforeCount(),afterCount() ); }
      
    private:
      inline IdLocalizationSerializeBuffer(size_type nBefore, size_type nAfter) : m_ids_before_count(nBefore), m_ids_after_count(nAfter) {}
      inline IdLocalizationSerializeBuffer() {}

      size_type m_ids_before_count;
      size_type m_ids_after_count;
      IdLocalization m_ids[0];
  };

  /*
    Helper class to serialize data movement info
  */
  class IdMoveSerializeBuffer
  {
    public:
      static inline size_type bufferSize( size_type N)
      {
          return sizeof(IdMoveSerializeBuffer) + N*sizeof(IdMove);
      }
      
      // WARNING: DO NOT delete objects allocated this way if allocPtr was not nullptr
      static inline IdMoveSerializeBuffer* alloc( size_type N, char* allocPtr = nullptr )
      {
          if( allocPtr == nullptr )
          {
              allocPtr = new char[ bufferSize(N) ];
          }
          return new( allocPtr ) IdMoveSerializeBuffer(N);
      }

      // WARNING: DO NOT delete objects bound to a memory location
      static inline IdMoveSerializeBuffer* bindTo( char* ptr )
      {
          return new( ptr ) IdMoveSerializeBuffer();
      }    
      
      inline size_type size() const { return bufferSize( count() ); }

      inline size_type count() const { return m_count; }
      inline IdMove& id_move(index_type i) { return m_idmove[i]; }
      
    private:
      inline IdMoveSerializeBuffer(size_type N) : m_count(N) {}
      inline IdMoveSerializeBuffer() {}

      size_type m_count;
      IdMove m_idmove[0];
  };

} // namespace XsDataMove
