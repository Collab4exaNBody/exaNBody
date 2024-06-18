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

#include<tuple>
#include <cstddef>

namespace exanb
{
	template<typename ItemType>
		struct ExtraDynamicStorageDataGridMoveBufferT;

	/**
	 * @brief Template struct representing a storage for extra dynamic data associated with cells in a move buffer.
	 * @tparam T The type of extra dynamic data stored in the cell move buffer.
	 */
	template <typename ItemType>
		struct ExtraDynamicDataStorageCellMoveBufferT
		{
      using InfoType = ExtraStorageInfo; 
      using UIntType = ExtraStorageInfo::UIntType;
			std::vector< size_t > m_indirection; /**< Indirection vector storing indices of the start of each particle's data in m_data. */
			std::vector< uint8_t > m_data; /**< Data vector storing the actual data for each cell, packed contiguously. */

			/**
			 * @brief Clears the extra dynamic data storage and resets global information.
			 * It also resets the global information in the data vector by resizing it and setting the number of particles and items to zero.
			 */
			inline void clear()
			{
				m_indirection.clear();
				m_data.clear();
				m_data.resize(2 * sizeof(UIntType)); // number of particles, number of items 
				UIntType* global_information = (UIntType*) m_data.data();
				global_information[0] = UIntType(0);
				global_information[1] = UIntType(0);
			}

			/**
			 * @brief Decodes the data storage in 3 pointers.
			 * @param n_particles The number of particles.
			 * @return A tuple containing const pointers to the global information, information tuple, and data vector.
			 */
			std::tuple<const UIntType*, const InfoType*, const ItemType*> decode_pointers(const unsigned int n_particles) const
			{
				const UIntType* glob_info_ptr = (UIntType*) m_data.data();
				const InfoType*      info_ptr = (InfoType*) (glob_info_ptr + 2);
				const ItemType*      data_ptr = (ItemType*) (info_ptr + n_particles);
				return {glob_info_ptr, info_ptr, data_ptr};
			}

			/**
			 * @brief Decodes the data storage in 3 pointers.
			 * @param n_particles The number of particles.
			 * @return A tuple containing const pointers to the global information, information tuple, and data vector.
			 */
			std::tuple<UIntType*, InfoType*, ItemType*> decode_pointers(const unsigned int n_particles) 
			{
				UIntType* glob_info_ptr = (UIntType*) m_data.data();
				InfoType*      info_ptr = (InfoType*) (glob_info_ptr + 2);
				ItemType*      data_ptr = (ItemType*) (info_ptr + n_particles);
				return {glob_info_ptr, info_ptr, data_ptr};
			}

			/**
			 * @brief Retrieves the information tuple for a specific particle.
			 * @param p The index of the particle.
			 * @return A reference to the information tuple for the specified particle.
			 */
			inline const InfoType& get_info(const unsigned int p) const
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(UIntType);
				const InfoType * __restrict__ info_ptr = (InfoType *) (m_data.data() + global_information_shift);
				return info_ptr[m_indirection[p]];
			}

			/**
			 * @brief Retrieves the information tuple for a specific particle.
			 * @param p The index of the particle.
			 * @return A reference to the information tuple for the specified particle.
			 */
			inline InfoType& get_info(const unsigned int p)
			{
				constexpr unsigned int global_information_shift = 2 * sizeof(UIntType);
				InfoType * __restrict__ info_ptr = (InfoType*) (m_data.data() + global_information_shift);
				return info_ptr[m_indirection[p]];
			}

			/**
			 * @brief Gets the total number of particles stored in the storage.
			 * @return The total number of particles.
			 */
			inline size_t number_of_particles() const
			{
#ifndef NDEBUG
				if( m_indirection.size() == 0 ) return 0;
				UIntType* glob_info_ptr = (UIntType*) m_data.data();
				UIntType np = glob_info_ptr[0];
				assert( m_indirection.size() == np );
#endif
				return m_indirection.size();
			}

			/**
			 * @brief Swaps the indirection values for two particles.
			 * @param a The index of the first particle.
			 * @param b The index of the second particle.
			 */
			inline void swap( size_t a, size_t b )
			{
				assert ( a < m_indirection.size() );
				assert ( b < m_indirection.size() );
				std::swap( m_indirection[a] , m_indirection[b] );
			}

			/**
			 * @brief Retrieves particle's id.
			 * @param p The index of the particle.
			 * @return The particle's id.
			 */
			inline uint64_t particle_id(size_t p) const
			{
				assert ( p < m_indirection.size() );
				auto& particle_information = this->get_info(p);
				return particle_information.pid;
			}

			/**
			 * @brief Retrieves the number of items associated with a particle.
			 * @param p The index of the particle.
			 * @return The number of items associated with the particle.
			 */
			inline UIntType particle_number_of_items(size_t p) const
			{
				assert ( p < m_indirection.size() );
				auto& particle_information = this->get_info(p);
				return particle_information.size;
			}

			/**
			 * @brief Retrieves the range of data associated with a particle.
			 * @param p The index of the particle.
			 * @return A pair of const pointers representing the beginning and end of the data range associated with the particle.
			 */
			inline std::pair< const ItemType* , const ItemType* > particle_data_range(size_t p) const
			{
				assert ( p < m_indirection.size() );
				constexpr unsigned int global_information_shift = 2 * sizeof(UIntType);
				UIntType n_particles = ((UIntType*) m_data.data()) [0]; // 0 -> n_particles, 1 -> n_items
				const auto [offset, size, id] = this->get_info(p);
				const ItemType* data_ptr = (const ItemType*) (m_data.data() + global_information_shift + n_particles * sizeof (InfoType)); 
				return { data_ptr + offset, data_ptr + offset + size };
				//return { data_ptr + offset, data_ptr + offset + size -1 };
			}

			/**
			 * @brief Retrieves a specific item associated with a particle.
			 * @param p The index of the particle.
			 * @param i The index of the item within the particle's data.
			 * @return A const pointer to the item associated with the particle at the specified index.
			 */
			inline const ItemType* item(size_t p, size_t i) const
			{
				assert ( p < m_indirection.size() );
				constexpr unsigned int global_information_shift = 2 * sizeof(UIntType);
				UIntType n_particles = this->number_of_particles(); // 0 -> n_particles, 1 -> n_items
				const auto [offset, size, id] = this->get_info(p);
				const ItemType* data_ptr = (const ItemType*) (m_data.data() + global_information_shift + n_particles * sizeof (InfoType)); 
				return data_ptr + offset + i;
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
				if( number_of_particles() == 0) return true; 
				const auto [glob_ptr, info_ptr, data_ptr] = decode_pointers(number_of_particles());
				UIntType n_particles = glob_ptr[0];
				return migration_test::check_info_consistency ( info_ptr, n_particles) ;
			}

      /** @brief test if the number of items per particle doesn't exceed 1e6.
			 * @return True the number of items per particle doesn't exceed 1e6, false otherwise.
       */
      inline bool check_info_value()
			{
				if( number_of_particles() == 0) return true; 
				constexpr UIntType limit = 1e6; // maximal number of item allowed for one particle.
        const auto [glob_ptr, info_ptr, data_ptr] = decode_pointers(number_of_particles());
        UIntType n_particles = glob_ptr[0];
				return migration_test::check_info_value ( info_ptr, n_particles, limit ); 
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

			/**
			 * @brief Copies incoming particle data from a ExtraDynamicStorageDataGridMoveBufferT into the current move buffer.
			 * @param opt_buffer The grid move buffer containing incoming particle data.
			 * @param cell_i The index of the cell of  the particle data copied.
			 * @param p_i The index of the particle within the cell.
			 */
			inline void copy_incoming_particle( const ExtraDynamicStorageDataGridMoveBufferT<ItemType>& opt_buffer, size_t cell_i, size_t p_i); 
		};

	/**
	 * @brief Template struct representing a grid of buffer for extra dynamic storage data in a grid.
	 * @tparam T The type of extra dynamic storage data.
	 */
	template<typename ItemType>
		struct ExtraDynamicStorageDataGridMoveBufferT
		{
      using InfoType = ExtraStorageInfo; 
      using UIntType = ExtraStorageInfo::UIntType;
			using CellMoveBuffer = ExtraDynamicDataStorageCellMoveBufferT<ItemType>;
			using CellStorage    = CellExtraDynamicDataStorageT<ItemType>;
			onika::memory::CudaMMVector< CellExtraDynamicDataStorageT<ItemType> > & m_cell_data;
			CellMoveBuffer & m_otb_buffer; /** buffer for elements going outside of grid */
			std::vector<CellMoveBuffer> m_cell_buffer; /** incoming buffers for elements entering another cell of the grid */
			inline CellMoveBuffer* otb_buffer() { return & m_otb_buffer; }
			inline CellMoveBuffer* cell_buffer( size_t cell_i ) { return & m_cell_buffer[cell_i]; }

			/**
			 * @brief Initializes the grid buffer with the specified number of cells.
			 * @param n_cells The number of cells to initialize the move buffer for.
			 */
			inline void initialize( size_t n_cells )
			{
				m_cell_buffer.resize( n_cells );
				for(size_t i = 0 ; i < n_cells ; i++) 
				{
					m_cell_buffer[i].clear();
				}
				m_otb_buffer.clear();
			}
			/**
			 * @brief Packs particles of a cell into a contiguous buffer.
			 * It performs the packing according to the same particle re-organization as in move_particles_across_cells.h.
			 * @param cell_i The index of the cell whose particles are to be packed.
			 * @param packed_particles Reference to a vector to store the packed particles.
			 * @param removed_particles Flag indicating whether to include removed particles in the packing. Default is true.
			 */
			//inline void pack_cell_particles( size_t cell_i, const std::vector<int32_t> & packed_particles, bool removed_particles = true )
			inline void pack_cell_particles( size_t cell_i, const std::vector<int> & packed_particles, bool removed_particles = true )
			{
				assert( cell_i < m_cell_data.size() );
				auto & cell = m_cell_data[cell_i];

				const size_t n_incoming_particles = m_cell_buffer[cell_i].number_of_particles();
				const size_t total_particles = removed_particles ? ( packed_particles.size() + n_incoming_particles ) : ( cell.number_of_particles() + n_incoming_particles );

				CellStorage pf;
				pf.initialize( total_particles );

				// 1. pack existing partciles, overwritting removed ones
				size_t p = 0;
				UIntType offset = 0;
				if( removed_particles )
				{
					const size_t n_particles = packed_particles.size();
					for( p = 0 ; p < n_particles ; p++)
					{
						// update information -> (offset , number of items, particle id) 
						const uint64_t id = cell.particle_id( packed_particles[p] );
						const UIntType n_items = cell.particle_number_of_items( packed_particles[p] );
					  assert( n_items < UIntType(1e6) ); // safe guard
						auto& particle_information = pf.m_info[p];
						particle_information = {offset, n_items, id};
						// update data
						pf.m_data.resize (offset + n_items);
						for(size_t i = 0 ; i < n_items ; i++)
						{
							pf.set_item( offset++, cell.get_particle_item( packed_particles[p] , i ) );
						}
					}
				}
				else // copy original particles as is
				{
					const size_t n_particles = cell.number_of_particles();
					for( p = 0 ; p < n_particles ; p++)
					{
						// update information
						const uint64_t id = cell.particle_id( p );
						const UIntType n_items = cell.particle_number_of_items( p );
					  assert( n_items < UIntType(1e6) ); // safe guard
						auto& particle_information = pf.m_info[p];
						particle_information = {offset, n_items, id};
						// update data
						pf.m_data.resize (offset + n_items);
						for(size_t i = 0 ; i < n_items ; i++)
						{
							pf.set_item( offset++, cell.get_particle_item( p , i ));
						}
					}
				}

				// add new items in the futur cell extra data storage
				// offset and p are incremented in the last loop 
				for(size_t i = 0 ; i < n_incoming_particles ; i++)
				{
					// update information -> (offset , number of items, particle id) 
					const uint64_t id = m_cell_buffer[cell_i].particle_id(i);
					const UIntType n_items = m_cell_buffer[cell_i].particle_number_of_items(i);
					assert( n_items < UIntType(1e6) ); // safe guard
					auto& particle_information = pf.m_info[p];
					particle_information = {offset, n_items, id};
					// update data
					pf.m_data.resize (offset + n_items);
					for(size_t j = 0 ; j < n_items ; j++) 
					{
						pf.set_item( offset++, *(m_cell_buffer[cell_i].item(i,j)) );
					}
					++ p;
				}

				assert( p == total_particles );
				assert ( pf.check_info_consistency() );
			  assert ( migration_test::check_info_value( pf.m_info.data(), pf.m_info.size() , 1e6 )); // check number of item
				// copy ad fit new dynamic extra data in the cell
				cell = std::move( pf );
				cell.shrink_to_fit();
			}

      inline bool check()
			{
#ifndef NDEBUG
//				// Warning, the error message is displayed if the mpi process is 0
				for (auto& it : m_cell_data ) 
				{	
					if(!it.check())
					{
						std::cout << "CHECK= Error in m_cell_data" << std::endl; 
						return false;
					}
				}
				if(!m_otb_buffer.check())
				{
					std::cout << "CHECK= Error in m_otb_buffer" << std::endl;
					return false;
				}
				for (auto& it : m_cell_buffer ) 
				{
					if(!it.check())
					{
						std::cout << "CHECK= Error in m_cell_buffer" << std::endl;
						return false;
					}
				}
				return true;
#else
				return true;
#endif
			}
		};

	/**
	 * @brief Copies incoming particle data from a move buffer into the cell.
	 * @tparam T The type of the extra data storage in the cell.
	 * @param opt_buffer The move buffer containing incoming particle data.
	 * @param cell_i The index of the cell to which the incoming particle data is copied.
	 * @param p_i The index of the particle within the cell.
	 */
	template<typename ItemType>
		inline void ExtraDynamicDataStorageCellMoveBufferT<ItemType>::copy_incoming_particle( const ExtraDynamicStorageDataGridMoveBufferT<ItemType>& opt_buffer, size_t cell_i, size_t p_i)
		{
			// Note : This function is not optimal, it does one std::move for every calls.
			// get cell information
			const CellExtraDynamicDataStorageT<ItemType>& cell = opt_buffer.m_cell_data[cell_i];
			const auto [offset, size, id] = cell.m_info[p_i];
			UIntType n_items_to_copy = size;
			assert ( n_items_to_copy <= UIntType(1e6) ); // less than 1M of new item
			assert ( migration_test :: check_info_consistency (cell.m_info.data(), cell.m_info.size()) );

			// reminder : buffer layout : N/nb particles, M/nb items, info1, info2 ..., infoN, item1, ..., itemM
			constexpr UIntType global_information_shift = 2 * sizeof(UIntType);
			UIntType* global_information_ptr = (UIntType*) m_data.data();
			const UIntType buffer_n_particles = global_information_ptr [0]; // 0 -> n_particles, 1 -> n_items
			const UIntType buffer_n_items     = global_information_ptr [1]; // 0 -> n_particles, 1 -> n_items

			// create shift to insert a new info
			// -> new buffer size
			const UIntType new_size = global_information_shift + (buffer_n_particles + 1) * sizeof(InfoType) + (buffer_n_items + n_items_to_copy) * sizeof(ItemType);

			// -> resize data with new size
			m_data.resize(new_size); // m_data ptr can change here

			// update pointers due to resize
			global_information_ptr = (UIntType*) m_data.data();
			InfoType * info_ptr = (InfoType*) (m_data.data() + global_information_shift);

			// -> shift item data and new info
			uint8_t* old_item_ptr = m_data.data() + global_information_shift + buffer_n_particles * sizeof(InfoType);
			uint8_t* new_item_ptr = m_data.data() + global_information_shift + (buffer_n_particles + 1) * sizeof(InfoType);
			UIntType nb_bytes_n_items = buffer_n_items * sizeof(ItemType);

			// --> create a space for info
			std::move(old_item_ptr, old_item_ptr + nb_bytes_n_items, new_item_ptr);

			// --> add info in this the new memory space
			if( buffer_n_particles != 0 )
			{
				InfoType& old_last = info_ptr[buffer_n_particles - 1];
				InfoType& new_last = info_ptr[buffer_n_particles];
				new_last = {old_last.offset + old_last.size, n_items_to_copy, id};
			}
			else
			{
				info_ptr[0] = {0, n_items_to_copy, id};
			}
			assert ( migration_test::check_info_value( info_ptr, buffer_n_particles , 1e6 )); // check number of item
			assert ( check_info_consistency () );

			// copy items at the end
			ItemType* item_ptr = (ItemType*) new_item_ptr;
			std::copy(
					cell.m_data.data() + offset, // offset and size are defined at the begining 
					cell.m_data.data() + offset + size,
					item_ptr + buffer_n_items // do not forget to shift here -> buffer_n_items is the old number of items on the buffer
					);				

			// update global information
			global_information_ptr[0] += 1;
			global_information_ptr[1] += n_items_to_copy;

			// update indirection vector
			m_indirection.push_back(m_indirection.size());
			assert ( m_indirection.size() == global_information_ptr[0] );
		}
}
