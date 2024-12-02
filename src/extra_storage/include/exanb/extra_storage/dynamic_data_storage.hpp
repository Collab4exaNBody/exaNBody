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

#include <exanb/core/grid.h>
#include <memory>
#include <exanb/extra_storage/extra_storage_info.hpp>
#include <exanb/extra_storage/migration_test.hpp>
#include <vector>
#include <cstddef>

namespace exanb
{
  using namespace std;
  using InfoType = ExtraStorageInfo; //std::tuple<UIntType,UIntType, UIntType>;
  using UIntType = ExtraStorageInfo::UIntType;

  /**
   * @brief Decodes the data storage in 3 pointers.
   * @param n_particles The number of particles.
   * @return A tuple containing const pointers to the global information, information tuple, and data vector.
   */
  template<typename ItemType>
  std::tuple<UIntType*, InfoType*, ItemType*> decode_pointers(void* buffer, const unsigned int n_particles)
  {
    UIntType* glob_info_ptr = (UIntType*) buffer;
    InfoType*      info_ptr = (InfoType*) (glob_info_ptr + 2);
    ItemType*      data_ptr = (ItemType*) (info_ptr + n_particles);
    return {glob_info_ptr, info_ptr, data_ptr};
  }

  /**
   * @brief Template struct representing the storage for extra dynamic data associated with cells.
   *
   * This template struct defines the storage for extra dynamic data associated with cells.
   * It consists of an offset vector and a data vector.
   *
   * @tparam T The type of extra dynamic data stored in the cell storage.
   */
  template<typename ItemType> struct CellExtraDynamicDataStorageT
  {
    //template <typename T> using VectorT = onika::memory::CudaMMVector<T>; // Warning, as we're using the classifier, we don't need this data to be on the gpu. 
    template <typename T> using VectorT = std::vector<T>;
    VectorT<InfoType> m_info; /**< Info vector storing indices of the [start, number of items, particle id] of each cell's extra dynamic data in m_data. */ 
    VectorT<ItemType> m_data; /**< Data vector storing the extra dynamic data for each cell. */

    /**
     * @brief Gets the total number of particles stored in the data structure.
     * This function returns the total number of particles stored in the data structure.
     * @return The total number of particles.
     */
    inline size_t number_of_particles() const
    {
      return m_info.size();
    }

    /**
     * @brief Gets the number of items associated associated to a specific particle.
     * This function returns the number of items associated with the particle specified by index 'p'.
     * @param p The index of the particle in the cell.
     * @return The number of items associated with the particle.
     */
    inline UIntType particle_number_of_items( size_t p ) const
    {
      return m_info[p].size;
    }

    /**
     * @brief Adds a new range of Extra Storage type.
     * This function appends a new particle with its associated data, provided as a std::vector 'ppf',
     * to the data structure. It updates the offset vector and appends the data to the data vector.
     * @param ppf The std::vector containing data associated with the new particle.
     * @param id The particle ID
     */
    inline void push_back( const UIntType id, const std::vector<ItemType>& ppf )
    {
      if(  m_info.size() == 0 )
      {
        InfoType new_info = {0, ppf.size(), id};
        m_info.push_back( new_info );
      }
      else
      {
        const auto & last = m_info.back();
        const unsigned int offset = last.offset + last.size;
        InfoType new_info = {offset, ppf.size(), id};
        m_info.push_back( new_info );
      }
      m_data.push_back(ppf.begin(), ppf.end());
    }

    inline void set_item(const UIntType offset, const ItemType& item )
    {
      m_data[offset] = item;
    }

    /**
     * @brief Copy a range in Extra Storage type.
     */
    inline void set_data_storage( const std::vector<ItemType>& ppf )
    {
      m_data.resize( ppf.size() );
      std::copy ( ppf.begin(), ppf.end(), m_data.begin());
    }

    /**
     * @brief Initializes the data structure for a given number of particles.
     *
     * This function initializes the data structure for a given number of particles.
     * If the number of particles is 0, it clears both the offset and data vectors.
     * Otherwise, it reserves memory for the offset vector, sets the initial offset value,
     * and resizes the data vector to 0.
     *
     * @param n_particles The number of particles to initialize the data structure for.
     */
    inline void initialize ( const unsigned int n_particles)
    {
      m_info.clear();
      m_data.clear();

      if ( n_particles == 0 )
      {
        return;
      }

      m_info.resize (n_particles);
      //m_info.assign (n_particles, {0,0,0});
      //m_info.assign (n_particles, {0,0,UIntType(-1)});
    }

    inline void clear()
    {
      initialize(0);
    }

    /**
     * @brief Gets a pointer to the storage info data.
     * @return A const pointer to the storage info data.
     */
    inline const InfoType* storage_info_ptr() const // bytes
    {
      return m_info.data();
    }

    /**
     * @brief Gets a pointer to the storage info data.
     * @return A const pointer to the storage info data.
     */
    inline const InfoType* storage_info_ptr() // bytes
    {
      return m_info.data();
    }

    /**
     * @brief Shrinks the capacity of the data structure to fit its size.
     * This function shrinks the capacity of the info and data vectors to fit their respective sizes.
     */
    inline void shrink_to_fit()
    {
      m_info.shrink_to_fit();
      m_data.shrink_to_fit();
    }

    /**
     * @brief Computes the total storage size in bytes.
     * @return The total storage size in bytes.
     */
    inline size_t storage_size() const
    {
      return m_data.size() * sizeof(ItemType) + m_info.size() * sizeof(InfoType);
    }

    /**
     * @brief Encodes the cell structure data to a buffer.
     * This function encodes the cell structure data (offsets and data) to a buffer.
     * It copies the offset and data vectors into the buffer sequentially.
     * @param buffer A pointer to the buffer where the cell structure data will be encoded.
     */
    inline void encode_cell_to_buffer ( void* buffer )
    {
      // get size in bytes
      const size_t info_size = m_info.size() * sizeof(InfoType);
      const size_t data_size = m_data.size() * sizeof(ItemType);

      const uint8_t * const __restrict__ info_ptr   = (const uint8_t*) onika::cuda::vector_data( m_info );
      const uint8_t * const __restrict__ data_ptr   = (const uint8_t*) onika::cuda::vector_data( m_data );
      uint8_t * __restrict__ buffer_ptr = (uint8_t*) buffer;

      // cast ptr in UIntType to store the number of particles and items
      UIntType * __restrict__ buffer_ptr_global_info = (UIntType*) buffer_ptr;
      buffer_ptr_global_info[0] = m_info.size();
      buffer_ptr_global_info[1] = m_data.size();

      const unsigned int global_info_shift = 2 * sizeof(UIntType);

      assert ( migration_test::check_info_consistency( m_info.data(), m_info.size() ));  
      assert ( migration_test::check_info_value( m_info.data(), m_info.size() , 1e6 )); // check number of item

      // first offset, then type T
      std::copy ( info_ptr, info_ptr + info_size, buffer_ptr + global_info_shift );
      std::copy ( data_ptr, data_ptr + data_size, buffer_ptr + global_info_shift + info_size );

      // re-check
      [[maybe_unused]] InfoType * __restrict__ check_ptr = (InfoType*)(buffer_ptr + global_info_shift);
      assert ( migration_test::check_info_value( check_ptr, m_info.size() , 1e6  )); // check number of item
    }

    /**
     * @brief Decodes the buffer data into the Extra Data Storage.
     * @param buffer A pointer to the buffer containing the encoded cell structure data.
     */
    inline void decode_buffer_to_cell ( void* buffer)
    {
      // cast ptr
      const UIntType* buff_ptr = (const UIntType*) buffer;

      // first two variables contains the number of particles and the second one contains the number of items (it could be deduced from offset)
      UIntType n_particles = buff_ptr [0];
      UIntType n_items     = buff_ptr [1];

      if ( n_particles == 0 && m_info.size() == 0)
      {
        this->clear();
        return;
      }
      //auto& last = m_info.size() > 0 ? m_info.back() : {0,0,0};   // only used in the next line    
      // resize data
      m_info.resize(n_particles);
      m_data.resize(n_items);
      const UIntType info_size = m_info.size();
      const UIntType data_size = m_data.size(); // std::get<0> (last) + std::get<1> (last);

      // get data pointers
      InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
      ItemType * const __restrict__ data_ptr = onika::cuda::vector_data( m_data );

      // get buffer pointers
      auto [buff_global, buff_info_ptr, buff_data_ptr] = decode_pointers<ItemType> (buffer, n_particles);

      // Some checks
      assert ( migration_test::check_info_consistency( buff_info_ptr, info_size));  
      assert ( migration_test::check_info_value( buff_info_ptr, info_size , 1e6 )); // check number of item

      // first informaions, then items
      std::copy ( buff_info_ptr, buff_info_ptr + info_size, info_ptr );
      std::copy ( buff_data_ptr, buff_data_ptr + data_size, data_ptr );
    }

    inline void append_data_stream_range (const uint8_t* buffer, size_t start, size_t end)
    {
/*
      if( start == 0 && end == 0 && this->number_of_particles() == 0 )
      {
        m_data.clear();
        return;
      }
*/
      if ( start >= end ) return;

      // resize info vector to add new infos
      size_t old_info_size = this->number_of_particles();
      size_t new_info_size = old_info_size + end - start;
      m_info.resize(new_info_size);

      // compute shift pointers for info and data vectors
      const UIntType * const buff      = ( UIntType*) buffer;
      const UIntType buff_n_particles = buff[0];
      [[maybe_unused]] const UIntType buff_n_items = buff[1];
  
    // get buffer pointers
      const auto [buff_global, buff_info, buff_data] = decode_pointers<ItemType> ((void*)buffer, buff_n_particles);
      assert ( migration_test::check_info_value( buff_info, buff_n_particles, 1e6) && "too many items for one particle, error"); // check the number of items per info

      // copy new information and update offset to fit with the current cell extra data storage
      // sizes and ids do not change
      std::copy ( buff_info + start, buff_info + end , m_info.data() + old_info_size);
      for(size_t i = 0 ; i < (end - start) ; i++)
      {
        const size_t idx = old_info_size + i;
        if(idx == 0) m_info[idx].offset = 0;
        else // idx > 0
        {
          const size_t lastIdx = idx - 1;
          const auto [last_offset, last_size, id] = m_info[lastIdx];
          m_info[idx].offset = last_offset + last_size;
        }
      }

      assert ( migration_test::check_info_doublon    ( m_info.data(), m_info.size() ));
      assert ( migration_test::check_info_consistency( m_info.data(), m_info.size() ));

      // define item range [first, last] | note: last particle idx = end-1 
      UIntType first_item = buff_info[start].offset;
      UIntType last_item  = buff_info[end - 1].offset + buff_info[end-1].size;
      UIntType new_items_to_append = last_item - first_item;

      //if( buff_n_items != new_items_to_append ) std::cout << buff_n_items << " != " << new_items_to_append  << " start " << start << "end " << end << " values " << buff_info[start].offset << " " << buff_info[end - 1].offset   << " " << buff_info[end-1].size << std::endl;
      assert ( buff_n_items >= new_items_to_append );

      if( new_items_to_append == 0 ) return;       

      const size_t old_data_size = m_data.size();

      // now resize data and copy new data in this memory place
      m_data.resize(old_data_size + new_items_to_append);
      std::copy ( buff_data + first_item , buff_data + last_item , m_data.data() + old_data_size);  
    }

    /**
     * @brief Retrieves the particle data and its size for a given particle index.
     * @param p The index of the particle.
     * @return A tuple containing a pointer to the particle data and the size of the data.
     */
    ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id(const unsigned int p) const
    {
      const InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
      return info_ptr[p].pid;
    }

    /**
     * @brief Retrieves the particle data and its size for a given particle index.
     * @param p The index of the particle.
     * @return A tuple containing a pointer to the particle data and the size of the data.
     */
    ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id(const unsigned int p) 
    {
      InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
      return info_ptr[p].pid;
    }


    /**
     * @brief Retrieves a specific item of a particle's data.
     * @param p The index of the particle.
     * @param n The index of the item within the particle's data.
     * @return A reference to the requested item of the particle's data.
     */
    ONIKA_HOST_DEVICE_FUNC inline ItemType& get_particle_item(const unsigned int p, const unsigned int n) const
    {
      const ItemType * const __restrict__ data_ptr = onika::cuda::vector_data( m_data );
      const InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
      const auto [offset, size, id ] = info_ptr[p]; 
      const ItemType * __restrict__ ppf_base = data_ptr + offset ;
      return ppf_base[ n ];
    }

    /**
     * @brief Retrieves a specific item of a particle's data.
     * @param p The index of the particle.
     * @param n The index of the item within the particle's data.
     * @return A reference to the requested item of the particle's data.
     */
    ONIKA_HOST_DEVICE_FUNC inline ItemType& get_particle_item(const unsigned int p, const unsigned int n)
    {
      ItemType * const __restrict__ data_ptr = onika::cuda::vector_data( m_data );
      const InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
      const auto [offset, size, id ] = info_ptr[p]; 
      ItemType * __restrict__ ppf_base = data_ptr + offset ;
      return ppf_base[ n ];
    }

    /**
     * @brief Compresses data using a specified function.
     * @tparam Func The type of the function object used for saving the compressed data.
     * @param save_data This functor object is responsible for saving the compressed data, it returns true if the data has to be saved.
     */
    template<typename Func>
      inline void compress_data(const Func& save_data)
      {
        if(m_info.size() == 0) return;
        if(m_data.size() == 0) return;
        // compress for each particle
        UIntType cur_off = 0;
        size_t itData = 0;
        // this function do not conserve the same data order per particle.
        for(size_t i = 0 ; i < m_info.size() ; i++)
        {
          UIntType acc = 0;
          size_t first_item = m_info[i].offset;
          size_t last_item = m_info[i].offset + m_info[i].size;
          for (size_t it = first_item ; it < last_item ; it++)
          {
            assert ( it < m_data.size() );
            if ( save_data( m_data[it] ) )
            {
              assert ( itData <= it );
              //m_data[itData++] = std::move(m_data[it]);
              m_data[itData++] = m_data[it];
              acc++;
            }
          }
          // update new information
          m_info[i].offset = cur_off;
          m_info[i].size = acc;
          cur_off += acc; 
        }

        m_data.resize(itData);

        // some tests
        check_info_consistency();
        check_info_value();
/*
        [[maybe_unused]] auto [last_offset, last_size, last_id] = m_info.back();
        assert ( itData == last_offset + last_size );
*/
      }

    /**
     * @brief Checks the consistency of the information stored in the storage.
     * This function checks the consistency of the information stored in the storage.
     * It verifies whether the information is consistent across all particles and returns true if consistent,
     * indicating that the data is correctly structured and organized.
     * @return True if the information is consistent across all particles, false otherwise.
     */
    inline bool check_info_consistency()
    {
      return migration_test::check_info_consistency( m_info.data(), m_info.size() );  
    }

    /** @brief test if the number of items per particle doesn't exceed 1e6.
     * @return True the number of items per particle doesn't exceed 1e6, false otherwise.
     */
    inline bool check_info_value()
    {
      constexpr UIntType limit = 1e6; // maximal number of item allowed for one particle.
      return migration_test::check_info_value ( m_info.data(), m_info.size(), limit );
    }

    /** @brief active all tests availables.
     * @return True if the information is consistent across all particles and the number of items per particle doesn't exceed 1e6, false otherwise.
     */
    inline bool check()
    {
      bool consistency = this->check_info_consistency();
      bool value       = this->check_info_value();
      return consistency && value;
    }
  };


  /**
   * @brief Template struct representing grid of extra dynamic data storage.
   * @tparam T The type of extra data stored in each cell.
   */
  template<typename ItemType>
    struct GridExtraDynamicDataStorageT  
    {
      onika::memory::CudaMMVector< CellExtraDynamicDataStorageT< ItemType > > m_data; /**< Memory-managed vector storing extra dynamic data storage for each cell. */
      GridExtraDynamicDataStorageT() {};
    };
}
