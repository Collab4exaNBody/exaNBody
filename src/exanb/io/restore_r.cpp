#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/backup_r.h>
#include <exanb/core/domain.h>

namespace exanb
{

  template<typename GridT>
  struct PositionRestoreNode : public OperatorNode
  {

    ADD_SLOT( GridT        , grid     , INPUT_OUTPUT );
    ADD_SLOT( Domain       , domain     , INPUT );
    ADD_SLOT( PositionBackupData , backup_r , INPUT);

    inline void execute ()  override final
    {
      IJK dims = grid->dimension();
      auto cells = grid->cells();
      const double cell_size = domain->cell_size();
      assert( backup_r->m_data.size() == grid->number_of_cells() );

      Mat3d mat;
      if( domain->xform() == backup_r->m_xform )
      {
        mat = make_identity_matrix();
      }
      else
      {
        mat = domain->inv_xform() * backup_r->m_xform;
      }

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc)
        {
	        const size_t n_particles = cells[i].size();
	        assert( backup_r->m_data[i].size() == n_particles*3 );

	        const Vec3d cell_origin = grid->cell_position( loc );

          const uint32_t* rb = backup_r->m_data[i].data();
          auto* __restrict__ rx = cells[i][field::rx];
          auto* __restrict__ ry = cells[i][field::ry];
          auto* __restrict__ rz = cells[i][field::rz];

#         pragma omp simd
          for(size_t j=0;j<n_particles;j++)
          {
            Vec3d rg { restore_u32_double(rb[j*3+0],cell_origin.x,cell_size)
                     , restore_u32_double(rb[j*3+1],cell_origin.y,cell_size)
                     , restore_u32_double(rb[j*3+2],cell_origin.z,cell_size) };
            Vec3d r = mat * rg;
            rx[j] = r.x;
            ry[j] = r.y;
            rz[j] = r.z;
          }

        }
        GRID_OMP_FOR_END
      }
    }

  };
    
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "restore_r", make_grid_variant_operator< PositionRestoreNode > );
  }

}

