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

#include <unordered_map>
#include <cstdint>
#include <onika/thread.h>

namespace exanb
{

  /*
    Concurrent unordered map which allows for safe concurrent insertion, using locks,
    and safe concurent finds and accesses IF AND ONLY IF no insertion happen concurrently with finds/accesses (find/access is lock free)
    Nevertheless, one can render concurent insert() and safe_at() calls 
  */
  template<class MapType , size_t _NbMetaBuckets = ::exanb::max_threads_hint*2 > // NbMetaBuckets may not exceed 65536
  struct MultiThreadedConcurrentMap
  {
    using key_type = typename MapType::key_type;
    using mapped_type = typename MapType::mapped_type;
    using value_type = typename MapType::value_type;
    using iterator = typename MapType::iterator;
    using const_iterator = typename MapType::const_iterator;
    
    static inline constexpr size_t NbMetaBuckets = _NbMetaBuckets;
    
    static inline size_t meta_bucket( const key_type& k )
    {
      static constexpr uint64_t count16 = 1ull<<16;
      static constexpr uint64_t mask16 = count16 - 1;
      uint64_t h = std::hash<key_type>{}( k );
      h = ( h ^ (h>>16) ^ (h>>32) ^ (h>>48) ) & mask16; // reduces to 0-65535
      return std::min( ( h * NbMetaBuckets ) / count16 , NbMetaBuckets - 1 );
    }

    inline const MapType& meta_bucket_map( const key_type& k ) const
    {
      return m_meta_bucket[ meta_bucket(k) ];
    }
    
    inline auto insert( const value_type& p )
    {
      const size_t mb = meta_bucket( p.first );
      assert( mb < m_meta_bucket.size() );
      m_meta_bucket_locks[ mb ].lock();
      auto inserted = m_meta_bucket[ mb ].insert( p );
      m_meta_bucket_locks[ mb ].unlock();
      return inserted;
    }

    inline const_iterator find( const key_type& k ) const
    {
      const size_t mb = meta_bucket(k);
      assert( mb < m_meta_bucket.size() );
      auto it = m_meta_bucket[ mb ].find(k);
      if( it == m_meta_bucket[ mb ].end() ) return m_meta_bucket[0].end();
      else return it;
    }

    inline const_iterator end() const
    {
      return m_meta_bucket[0].end();
    }

    inline mapped_type safe_at( const key_type& k )
    {
      const size_t mb = meta_bucket(k);
      assert( mb < m_meta_bucket.size() );
      m_meta_bucket_locks[ mb ].lock();
      auto it = m_meta_bucket[ mb ].find(k);
      assert( it != m_meta_bucket[ mb ].end() );
      auto value = it->second;
      m_meta_bucket_locks[ mb ].unlock();
      return value;
    }

    // not thread safe if concurrently used with insertions
    inline mapped_type at( const key_type& k ) const
    {
      const size_t mb = meta_bucket(k);
      assert( mb < m_meta_bucket.size() );
      auto it = m_meta_bucket[ mb ].find(k);
      assert( it != m_meta_bucket[ mb ].end() );
      auto value = it->second;
      return value;
    }

    inline void clear()
    {
      for(auto& m:m_meta_bucket) m.clear();
    }

    inline size_t size() const
    {
      size_t sz = 0;
      for(auto& m:m_meta_bucket) sz += m.size();
      return sz;
    }

    //inline void set_safe_concurrent_insert_at(bool yn) { m_safe_concurent_insertion_get = yn; }

    spin_mutex_array m_meta_bucket_locks { NbMetaBuckets };
    std::vector< MapType > m_meta_bucket { NbMetaBuckets };
  };

}
