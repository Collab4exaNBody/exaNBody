#pragma xstamp_cuda_enable

#pragma xstamp_grid_variant

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/defbox/push_vec3_2nd_order.h>
#include <exanb/defbox/push_vec3_2nd_order_xform.h>

#include <exanb/defbox/xform_mode.h>

namespace exanb
{

  template<
    class GridT,
    class Field_X, class Field_Y, class Field_Z,
    class Field_dX, class Field_dY, class Field_dZ,
    class Field_ddX, class Field_ddY, class Field_ddZ,
    class = AssertGridHasFields< GridT, Field_X, Field_Y, Field_Z, Field_dX, Field_dY, Field_dZ , Field_ddX, Field_ddY, Field_ddZ>
    >
  struct PushVec3SecondOrderXForm : public OperatorNode
  {
    using compute_field_set_t = FieldSet<Field_X, Field_Y, Field_Z, Field_dX, Field_dY, Field_dZ, Field_ddX, Field_ddY, Field_ddZ>;
    static constexpr compute_field_set_t compute_field_set {};
      
    /***************** Operator slots *********************/
    ADD_SLOT( GridT  , grid       , INPUT_OUTPUT);
    ADD_SLOT( double , dt         , INPUT , REQUIRED );
    ADD_SLOT( double , dt_scale   , INPUT , REQUIRED );
    ADD_SLOT( Domain , domain     , INPUT , REQUIRED );
    ADD_SLOT( XFormMode, xform_mode , INPUT , XFormMode::INV_XFORM );
    /******************************************************/

    inline void execute () override final
    {
      const double delta_t = (*dt) * (*dt_scale);
      const double delta_t2 = delta_t*delta_t*0.5;

      if( (*xform_mode) == XFormMode::IDENTITY || domain->xform_is_identity() )
      {
        ldbg<<"PushVec3SecondOrder: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode)<<std::endl;
        PushVec3SecondOrderFunctor func { delta_t , delta_t2 };
        compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
      }
      else
      {
        const Mat3d xform = (xform_mode->m_value==XFormMode::XFORM) ? domain->xform() : domain->inv_xform();
        ldbg<<"PushVec3SecondOrderXForm: dt="<<(*dt)<<", dt_scale="<<(*dt_scale)<<", xform_mode="<< (*xform_mode)<<", xform="<<xform<<std::endl;
        PushVec3SecondOrderXFormFunctor func { xform , delta_t , delta_t2 };
        compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
      }
    }

  };

  template<class GridT> using PushAccelVelocityToPositionXForm = PushVec3SecondOrderXForm<GridT, field::_rx,field::_ry,field::_rz,field::_vx,field::_vy,field::_vz,field::_fx,field::_fy,field::_fz>;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "push_f_v_r", make_grid_variant_operator< PushAccelVelocityToPositionXForm > );
  }

}

