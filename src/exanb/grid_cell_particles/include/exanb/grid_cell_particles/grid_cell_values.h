#pragma once

#include <vector>
#include <unordered_map>
#include <cassert>
#include <exanb/core/basic_types_def.h>
#include <exanb/core/grid_algorithm.h>
#include <onika/memory/allocator.h>

namespace exanb
{    
  struct GridCellField
  {
    size_t m_subdiv = 0;
    size_t m_components = 0; // equals m_subdiv*m_subdiv*m_subdiv * number of values per sub cell
    size_t m_offset = 0;
  };

  struct AddCellFieldInfo
  {
    std::string m_name;
    size_t m_subdiv=1;
    size_t m_comps=1;
  };

  template<class DataPtrT>
  struct GridCellFieldAccessor
  {
    DataPtrT m_data_ptr = nullptr;
    size_t m_stride = 0;
  };

  template<class T = double>
  struct GridCellValuesT
  {
    using GridCellValueType = T;
    using GridCellValueVector = onika::memory::CudaMMVector<GridCellValueType>;

    inline GridCellFieldAccessor<T*> field_data(const GridCellField& gcf)
    {
      return { m_data.data() + gcf.m_offset , m_components };
    }
    inline GridCellFieldAccessor<T*> field_data(const std::string& fname)
    {
      auto it = m_fields.find(fname);
      assert( it != m_fields.end() );
      return field_data( it->second );
    }
    
    inline GridCellFieldAccessor<const T*> field_data(const GridCellField& gcf) const
    {
      return { m_data.data() + gcf.m_offset , m_components };
    }
    inline GridCellFieldAccessor<const T *> field_data(const std::string& fname) const
    {
      auto it = m_fields.find(fname);
      assert( it != m_fields.end() );
      return field_data( it->second );
    }
    
    inline bool has_field(const std::string& fname) const { return m_fields.find(fname) != m_fields.end(); }
    inline const std::unordered_map< std::string , GridCellField >& fields() const { return m_fields; }
    inline const GridCellField& field(const std::string& fname) const { return m_fields.at(fname); }
    inline size_t number_of_cells() const { return grid_cell_count(m_grid_dims); }
    inline bool empty() const { return m_data.empty(); }
    inline size_t ghost_layers() const { return m_ghost_layers; }
    inline IJK grid_dims() const { return m_grid_dims; }
    inline IJK grid_offset() const { return m_grid_offset; }

    inline void add_fields(const std::vector<AddCellFieldInfo>& fields_to_add )
    {
      size_t values_per_cell = 0; //subdiv*subdiv*subdiv*comps;
      for(const auto& f : fields_to_add)
      {
        values_per_cell += f.m_subdiv * f.m_subdiv * f.m_subdiv * f.m_comps ;
      }
      size_t offset = m_components;
      size_t n_cells = number_of_cells();
      GridCellValueVector new_data( (m_components+values_per_cell) * n_cells , 0 );
      for(size_t i=0;i<n_cells;i++)
      {
        for(size_t j=0; j<m_components; j++)
        {
          new_data[ (m_components+values_per_cell)*i + j ] = m_data[ m_components*i + j ];
        }
      }
      m_data = std::move( new_data );
      m_components = offset + values_per_cell;
      for(const auto& f : fields_to_add)
      {
        size_t fvalues = f.m_subdiv * f.m_subdiv * f.m_subdiv * f.m_comps;
        m_fields.emplace( std::pair<std::string,GridCellField> { f.m_name , { f.m_subdiv,fvalues,offset } } );
        offset += fvalues;
      }
    }

    inline void add_field(const std::string& fname, size_t subdiv=1, size_t comps=1)
    {
      assert( m_fields.find(fname) == m_fields.end() );
      add_fields( { {fname,subdiv,comps} } );
    }
    
    inline size_t components() const { return m_components; }

    inline GridCellValueVector& data() { return m_data; }
    inline const GridCellValueVector& data() const { return m_data; }
    
    // if dims is different from this->m_grid_dims, then discards stored data (i.e., all values are set to 0 afterward)
    // else has no effect at all and values are unchanged
    inline void set_grid_dims( const IJK& dims )
    {
      m_grid_dims = dims;
      m_data.resize( number_of_cells() * m_components , 0.0 );
    }

    // informative, does impact storage in any way
    inline void set_ghost_layers( size_t gl )
    {
      m_ghost_layers = gl;
    }

    inline void set_grid_offset( const IJK& offset )
    {
      m_grid_offset = offset;
    }

    IJK m_grid_dims = { 0, 0, 0 };
    IJK m_grid_offset = { 0, 0, 0 };
    size_t m_ghost_layers = 0;
    size_t m_components = 0; // total number of values for each sub cell
    GridCellValueVector m_data;
    std::unordered_map< std::string , GridCellField > m_fields;
  };
  
  using GridCellValues = GridCellValuesT<>;

}

