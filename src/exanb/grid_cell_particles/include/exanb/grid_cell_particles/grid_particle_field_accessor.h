#pragma once

#include <onika/cuda/cuda.h>

namespace exanb
{

  template<class CellsT>
  struct GridParticleFieldAccessor
  {
    CellsT m_cells;
    
    template<class FieldIdT>
    ONIKA_HOST_DEVICE_FUNC inline typename FieldIdT::value_type get(size_t cell_i, size_t p_i, const FieldIdT& f ) const
    {
      return m_cells[cell_i][f][p_i];
    }
  };

} // end of namespace exanb

