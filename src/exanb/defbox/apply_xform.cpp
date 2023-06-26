#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/basic_types_operators.h>

#include <memory>

namespace exanb
{

  template< class GridT >
  class ApplyXForm : public OperatorNode
  {
  
    ADD_SLOT( GridT , grid , INPUT_OUTPUT);
    ADD_SLOT( Mat3d , xform , INPUT , REQUIRED );

  public:

    inline void execute ()  override final
    {
      const Mat3d mat = *xform;
      ldbg << "apply_xform: "<< mat << std::endl;
      auto cells = grid->cells();
      IJK dims = grid->dimension();
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN(dims,i,loc)
        {
          auto * __restrict__ rx = cells[i][field::rx]; ONIKA_ASSUME_ALIGNED(rx);
          auto * __restrict__ ry = cells[i][field::ry]; ONIKA_ASSUME_ALIGNED(ry);
          auto * __restrict__ rz = cells[i][field::rz]; ONIKA_ASSUME_ALIGNED(rz);
          size_t n = cells[i].size();
#         pragma omp simd
          for(size_t i=0;i<n;i++)
          {
            Vec3d d = mat * Vec3d{rx[i],ry[i],rz[i]};
            rx[i] = d.x;
            ry[i] = d.y;
            rz[i] = d.z;
          }
        }
        GRID_OMP_FOR_END
      }
    }

  };
  
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "apply_xform", make_grid_variant_operator< ApplyXForm > );
  }

}


