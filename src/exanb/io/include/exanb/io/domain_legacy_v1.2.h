#pragma once
#include <exanb/core/domain.h>

namespace exanb
{

  // Domain structure as it was prior to commit # 3f4b5697a70e263b6fd24f807515c3f2a7c3423b
  // which drastically changed how periodic, expandable and mirroring booleans are stored
  class alignas(8) Domain_legacy_v1_2
  {
  public:
    // grid_dimension() and m_grid_dims is the total size (in cells) of the whole simulation domain
    inline IJK grid_dimension() const { return m_grid_dims; }

    // bounds() is the whole simulation domain physical size
    inline AABB bounds() const { return m_bounds; }

    // cubic size of cell
    inline double cell_size() const { return m_cell_size; }

    // tells if particular directions are periodic or not
    inline bool periodic_boundary_x() const { return m_periodic[0]; }
    inline bool periodic_boundary_y() const { return m_periodic[1]; }
    inline bool periodic_boundary_z() const { return m_periodic[2]; }
    
    inline bool expandable() const { return m_expandable; }

    inline Mat3d xform() const { return m_xform; }

  private:
    AABB m_bounds { {0.,0.,0.} , {0.,0.,0.} };
    IJK m_grid_dims { 0, 0, 0 };
    double m_cell_size = 0.0;

    // transformation to the physical space
    Mat3d m_xform = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
    Mat3d m_inv_xform = { 1.,0.,0., 0.,1.,0., 0.,0.,1. };
    double m_xform_min_scale = 1.0;
    double m_xform_max_scale = 1.0;
    bool m_xform_is_identity = true;

    // boundary periodicity
    bool m_periodic[3] = {false,false,false};

    // expandable in the non periodic directions
    bool m_expandable = true;
  };

}

