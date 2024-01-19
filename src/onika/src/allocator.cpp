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
#include <iostream>
#include <malloc.h>

#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
#endif

#include <onika/memory/allocator.h>
#include <onika/debug.h>

#include <cstring>

namespace onika
{
  namespace memory
  {
  
#   ifdef ONIKA_CUDA_VERSION
    bool GenericHostAllocator::s_enable_cuda = true;
    bool GenericHostAllocator::cuda_enabled()
    {
      return s_enable_cuda;
    }
    void GenericHostAllocator::set_cuda_enabled(bool yn)
    {
      s_enable_cuda = yn;
    }
#   endif

#   ifndef NDEBUG
    bool GenericHostAllocator::s_enable_debug_log = false;
    void GenericHostAllocator::set_debug_log(bool b) { s_enable_debug_log = b; }
#   endif

    bool GenericHostAllocator::operator == (const GenericHostAllocator& other) const
    {
      return m_alloc_policy == other.m_alloc_policy;
    }
    
    HostAllocationPolicy GenericHostAllocator::get_policy() const
    {
      return cuda_enabled() ? m_alloc_policy : HostAllocationPolicy::MALLOC;
    }

    bool GenericHostAllocator::allocates_gpu_addressable() const
    {
      return get_policy() == HostAllocationPolicy::CUDA_HOST;
    }
    
    void GenericHostAllocator::set_gpu_addressable_allocation(bool yn )
    {
      m_alloc_policy = ( yn ? HostAllocationPolicy::CUDA_HOST : HostAllocationPolicy::MALLOC );
    }


    MemoryChunkInfo GenericHostAllocator::memory_info( void* ptr , size_t s ) const
    {
      MemoryChunkInfo info;
      info.alloc_base = ptr;
#     ifndef NDEBUG
      info.alloc_size = * reinterpret_cast<size_t*>( reinterpret_cast<uint8_t*>(ptr) + s );
      if( s != info.alloc_size )
      {
        std::cerr<<"Corrupted allocation trailer ("<<s<<"!="<<info.alloc_size<<")\n"<<std::flush;
        std::abort();
      }
      info.alloc_flags = * reinterpret_cast<uint32_t*>( reinterpret_cast<uint8_t*>(ptr) + s + sizeof(size_t) );
#     else
      info.alloc_flags = * reinterpret_cast<uint32_t*>( reinterpret_cast<uint8_t*>(ptr) + s );
#     endif

      auto mem_type = info.mem_type();
      unsigned int a = info.alignment();
      unsigned int a2=1; while(a2<a) a2*=2;
      bool flags_ok = ( mem_type==HostAllocationPolicy::CUDA_HOST || mem_type==HostAllocationPolicy::MALLOC ) && (a2==a) ;
      if( ! flags_ok )
      {
        std::cerr<<"GenericHostAllocator: memory chunk corrupted\n"<<std::flush;
        std::abort();
      }

      return info;
    }

    bool GenericHostAllocator::is_gpu_addressable( void* ptr , size_t s ) const
    {
      if( ptr == nullptr ) { return true; }
      auto mem_type = memory_info(ptr,s).mem_type();
      return (mem_type  == HostAllocationPolicy::CUDA_HOST );
    }

    void GenericHostAllocator::deallocate( void* ptr , size_t s ) const
    {
      if( ptr == nullptr )
      {
        assert( s == 0 );
        return;
      }
      assert( s > 0 );
      
      // general case, allocated size is known
      auto info = memory_info(ptr,s);
      switch( info.mem_type() )
      {
        case HostAllocationPolicy::MALLOC :
#         ifndef NDEBUG
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"MALLOC: free "<<info.size()<<" bytes @"<<info.base_ptr()<<" align="<<info.alignment()<<std::endl; }
#         endif
          free(info.alloc_base);
          break;
        case HostAllocationPolicy::CUDA_HOST :
#         ifndef NDEBUG
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"CUDA: free "<<info.size()<<" bytes @"<<info.base_ptr()<<" align="<<info.alignment()<<std::endl; }
#         endif
#         ifdef ONIKA_CUDA_VERSION
          // lout << "cudaFree(@"<<ptr<<","<<s<<") a="<<alignment<<std::endl;
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_FREE(info.alloc_base) );
#         else
          std::cerr << "Free memory with type CUDA_HOST but cuda is not available" << std::endl;
          std::abort();
#         endif
          break;
        default:
          std::cerr << "Corrupted memory flags (mem_type="<<(int)info.mem_type()<<")"<< std::endl;
          std::abort();
          break;
      }      
    }
  
    void* GenericHostAllocator::allocate(size_t s, size_t a) const
    {
      void* ptr = nullptr;
      auto alloc_pol = get_policy();
      switch( alloc_pol )
      {
        case HostAllocationPolicy::MALLOC :
        {
          a = std::max( a , sizeof(void*) ); // this is required by posix_memalign.
          int r = posix_memalign( &ptr, a, s + add_info_size );
          if( r != 0 ) { std::cerr<<"Allocation failed. aborting.\n"; std::abort(); }
#         ifndef NDEBUG
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"MALLOC: alloc "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @"<<ptr<< " , align="<<a<<std::endl; }
#         endif
        }
        break;

        case HostAllocationPolicy::CUDA_HOST :
        {
#         if defined(ONIKA_CUDA_VERSION)
          ptr = nullptr;
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC_MANAGED( &ptr, s + add_info_size ) );
          auto pa = reinterpret_cast<uint8_t*>(ptr) - (uint8_t*)nullptr;
          if( ( pa % a ) != 0 )
          {
            std::cerr << "cudaMallocManaged returned a pointer that is not aligned on a "<<a<<" bytes boundary"<<std::endl;
            std::abort();
          }          
#         ifndef NDEBUG
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"CUDA: alloc "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @"<<ptr<< " , align="<<a<<std::endl; }
#         endif
          // lout << "cudaMallocManaged("<<s<<","<<a<<") -> @"<<ptr<<std::endl;
#         else
          std::cerr << "Cuda is disabled, no support for CUDA_HOST allocation policy"<<std::endl;
          ptr = nullptr;
          std::abort();
#         endif
        }
        break;
        
        default:
        {
          std::cerr << "Corrupted allocation flag (unknown value "<<static_cast<uint32_t>(alloc_pol)<<")"<<std::endl;
          std::abort();
        }
        break;
      }

      if( s>0 && ptr==nullptr )
      {
        std::cerr<< "onika::memory::GenericHostAllocator::allocate("<<s<<","<<a<<") : Allocation failed (cuda_enabled="<<std::boolalpha<<cuda_enabled()<<")"<<std::endl<<std::flush;
        std::abort();
      }

#     ifdef ONIKA_MEMORY_ZERO_ALLOC
#     ifndef NDEBUG
      if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"zero "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @" << ptr << std::endl; }
#     endif
      if( ptr != nullptr ) { std::memset( ptr , 0 , s + add_info_size ); }
#     endif

      uint32_t alloc_flags = static_cast<uint32_t>(alloc_pol) | ( static_cast<uint32_t>(a) << 8 );
#     ifndef NDEBUG
      * reinterpret_cast<size_t*>( reinterpret_cast<uint8_t*>(ptr) + s ) = s;
      * reinterpret_cast<uint32_t*>( reinterpret_cast<uint8_t*>(ptr) + s + sizeof(size_t) ) = alloc_flags;
      const auto minfo = memory_info(ptr,s);
      assert( minfo.alloc_size == s );
      assert( minfo.alloc_flags == alloc_flags );
#     else
      * reinterpret_cast<uint32_t*>( reinterpret_cast<uint8_t*>(ptr) + s ) = alloc_flags;
#     endif
      
      return ptr;
    }


  }
}

