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
#include <exanb/extra_storage/migration_test.hpp>
#include <vector>
#include <cstddef>

namespace exanb
{
	using namespace std;
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
		using UIntType = uint64_t;
		using InfoType = std::tuple<UIntType,UIntType, UIntType>;
		onika::memory::CudaMMVector<InfoType> m_info; /**< Info vector storing indices of the [start, number of items, particle id] of each cell's extra dynamic data in m_data. */ 
		onika::memory::CudaMMVector<ItemType> m_data; /**< Data vector storing the extra dynamic data for each cell. */

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
		inline size_t particle_number_of_items( size_t p ) const
		{
			return std::get<1> (m_info[p]);
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
			if(	m_info.size() == 0 )
			{
				InfoType new_info = {0, ppf.size(), id};
				m_info.push_back( new_info );
			}
			else
			{
				const auto & last = m_info.back();
				const unsigned int offset = std::get<0> (last) + std::get<1> (last);
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
			m_info.assign (n_particles, {0,0,UIntType(-1)});
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

			const char * const __restrict__ info_ptr   = (const char*) onika::cuda::vector_data( m_info );
			const char * const __restrict__ data_ptr   = (const char*) onika::cuda::vector_data( m_data );
			char * __restrict__ buffer_ptr = (char*) buffer;

			// cast ptr in UIntType to store the number of particles and items
			UIntType * __restrict__ buffer_ptr_global_info = (UIntType*) buffer_ptr;
			buffer_ptr_global_info[0] = m_info.size();
			buffer_ptr_global_info[1] = m_data.size();

			const unsigned int global_info_shift = 2 * sizeof(UIntType);

			assert ( migration_test::check_info_consistency( m_info.data(), m_info.size() ));  

			// first offset, then type T
			std::copy ( info_ptr, info_ptr + info_size, buffer_ptr + global_info_shift );
			std::copy ( data_ptr, data_ptr + data_size, buffer_ptr + global_info_shift + info_size );
		}

		/**
		 * @brief Decodes the buffer data into the Extra Data Storage.
		 * @param buffer A pointer to the buffer containing the encoded cell structure data.
		 */
		inline void decode_buffer_to_cell ( void* buffer)
		{
			// cast ptr
			const char* buff_ptr = (const char *) buffer;

			// first two variables contains the number of particles and the second one contains the number of items (it could be deduced from offset)
			UIntType n_particles = ((UIntType *) buffer) [0];
			UIntType n_items     = ((UIntType *) buffer) [1];

			if ( n_particles == 0 && m_info.size() == 0)
			{
				this->clear();
				return;
			}
			//auto& last = m_info.size() > 0 ? m_info.back() : {0,0,0};	 // only used in the next line		
			// resize data
			m_info.resize(n_particles);
			m_data.resize(n_items);
			const UIntType info_size = m_info.size();
			const UIntType data_size = m_data.size(); // std::get<0> (last) + std::get<1> (last);

			// get data pointers
			InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
			ItemType * const __restrict__ data_ptr = onika::cuda::vector_data( m_data );


			// get buffer pointers
			const UIntType first_info = 2 * sizeof(UIntType);
			const UIntType first_data = first_info + info_size * sizeof(InfoType); 
			const InfoType * const __restrict__ buff_info_ptr = (const InfoType*) (buff_ptr + first_info);
			const ItemType * const __restrict__ buff_data_ptr = (const ItemType*) (buff_ptr + first_data);

			// first informaions, then items
			std::copy ( buff_info_ptr, buff_info_ptr + info_size, info_ptr );
			std::copy ( buff_data_ptr, buff_data_ptr + data_size, data_ptr );
		}

		inline void append_data_stream_range (const void* buffer, size_t start, size_t end)
		{
			if( start == 0 && end == 0 && this->number_of_particles() == 0 )
			{
				m_data.clear();
				return;
			}

			if ( start >= end ) return;

			// resize info vector to add new infos
			size_t old_info_size = this->number_of_particles();
			size_t new_info_size = old_info_size + end - start;
			m_info.resize(new_info_size);

			// compute shift pointers for info and data vectors
			 UIntType * const __restrict__ buff      = ( UIntType*) buffer;
			 UIntType buff_n_particles = buff[0];
			 InfoType * const __restrict__ buff_info = ( InfoType*) (buff + 2);
			 ItemType * const __restrict__ buff_data = ( ItemType*) (buff_info + buff_n_particles);

			for(size_t i = 0 ; i < buff[0] ; i++)
			{
				if(i == 0) std::get<0> (buff_info[i]) = 0;
				else
				{
					const size_t lastIdx = i - 1;
					const auto [last_offset, last_size, id] = buff_info[lastIdx];
					std::get<0> (buff_info[i]) = last_offset + last_size;
				}
			}
			assert ( migration_test::check_info_doublon    ( buff_info, buff_n_particles));
			assert ( migration_test::check_info_consistency( buff_info, buff_n_particles));

			// copy new information and update offset to fit with the current cell extra data storage
			// sizes and ids do not change
			std::copy ( buff_info + start, buff_info + end , m_info.data() + old_info_size);
			for(size_t i = 0 ; i < end - start ; i++)
			{
				if(old_info_size + i == 0) std::get<0> (m_info[old_info_size + i]) = 0;
				else
				{
					const size_t lastIdx = old_info_size + i - 1;
					const auto [last_offset, last_size, id] = m_info[lastIdx];
					std::get<0> (m_info[old_info_size + i]) = last_offset + last_size;
				}
			}

			assert ( migration_test::check_info_doublon    ( m_info.data(), m_info.size() ));
			assert ( migration_test::check_info_consistency( m_info.data(), m_info.size() ));

			// define item range [first, last] | note: last particle idx = end-1 
			UIntType first_item = std::get<0> (buff_info[start]);
			UIntType last_item  = std::get<0> (buff_info[end -1] ) + std::get<1> (buff_info[end-1]);
			UIntType new_items_to_append = last_item - first_item;

			if( new_items_to_append == 0 ) return;			 

			size_t old_data_size = m_data.size();

			// now resize data and copy new data in this memory place 
			m_data.resize(m_data.size() + new_items_to_append);
			std::copy ( buff_data + first_item , buff_data + last_item , m_data.data() + old_data_size);	
		}

		/**
		 * @brief Retrieves the particle data and its size for a given particle index.
		 * @param p The index of the particle.
		 * @return A tuple containing a pointer to the particle data and the size of the data.
		 */
		ONIKA_HOST_DEVICE_FUNC inline UIntType particle_id(const unsigned int p) const
		{
			const ItemType * const __restrict__ data_ptr = onika::cuda::vector_data( m_data );
			const InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
			return std::get<2> (info_ptr[p]);
		}

		/**
		 * @brief Retrieves the particle data and its size for a given particle index.
		 * @param p The index of the particle.
		 * @return A tuple containing a pointer to the particle data and the size of the data.
		 */
		ONIKA_HOST_DEVICE_FUNC inline UIntType particle_id(const unsigned int p) 
		{
			InfoType * const __restrict__ info_ptr = onika::cuda::vector_data( m_info );
			return std::get<2> (info_ptr[p]);
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
				if(m_data.size() == 0) return;
				// compress for each particle
				UIntType cur_off = 0;
				int itData = 0;
				// this function do not conserve the same data order per particle.
				for(size_t i = 0 ; i < m_info.size() ; i++)
				{
					auto& [offset, size, id] = m_info[i];
					UIntType acc = 0;

					size_t last_item = offset + size;
					size_t first_item = offset;

					for (size_t it = first_item ; it < last_item ; it++)
					{
						assert ( it < m_data.size() );
						if ( save_data( m_data[it] ) )
						{
							assert ( itData <= it );
							m_data[itData++] = std::move(m_data[it]);
							//m_data[itData++] = std::move(m_data[it]);
							acc++;
						}
					}
					// update new information
					offset = cur_off;
					size = acc;
					cur_off += acc;

				}

				m_data.resize(itData);

				// some tests
				[[maybe_unused]] auto [last_offset, last_size, last_id] = m_info.back();
				assert ( itData == last_offset + last_size );
				assert ( check_info_consistency() );	
			}


		inline bool check_info_consistency()
		{
			return migration_test::check_info_consistency( m_info.data(), m_info.size() );  
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
