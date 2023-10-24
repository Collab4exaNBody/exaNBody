//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <onika/memory/allocator.h>

#include <onika/soatl/field_pointer_tuple.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_cell_projection.h>

#include <memory>

namespace exanb
{

  template<class GridT>
  struct ParticleFluidFrictionFunctor
  {
    const GridT& grid;
    const double * __restrict__ cell_value_ptr = nullptr;
    const size_t cell_value_stride = 0;
    const unsigned int subdiv = 0;
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () (
      double rx, double ry, double rz,
      double vx, double vy, double vz,
      double& fx, double& fy, double& fz ) const
    {
      using namespace ParticleCellProjectionTools;

      assert( cell_value_stride == subdiv * 3 ); // we only accept 3d vector fields
      const Vec3d r = {rx,ry,rz};
      const Vec3d v = {vx,vy,vz};
      if( cell_value_ptr != nullptr )
      {
        const IJK dims = grid.dimension();
        const IJK loc = grid.locate_cell(r);
        const Vec3d cell_origin = grid.cell_position( loc );
        IJK center_cell_loc;
        IJK center_subcell_loc;
        Vec3d rco = r - cell_origin;
        const double cell_size = grid.cell_size();
        const double subcell_size = cell_size / subdiv;
        localize_subcell( rco, cell_size, subcell_size, subdiv, center_cell_loc, center_subcell_loc );
        center_cell_loc += loc;
        const ssize_t nbh_cell_i = grid_ijk_to_index( dims , center_cell_loc );
        const ssize_t nbh_subcell_i = grid_ijk_to_index( IJK{subdiv,subdiv,subdiv} , center_subcell_loc );
        assert( nbh_cell_i>=0 && nbh_cell_i<ssize_t(grid.number_of_cells()) );
        assert( nbh_subcell_i>=0 && nbh_subcell_i<(subdiv*subdiv*subdiv) );

        // access fluid velocity in the grid's sub cell at particle location
        const Vec3d flow_velocity = {
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*3 + 0 ] ,
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*3 + 1 ] ,
          cell_value_ptr[ nbh_cell_i * cell_value_stride + nbh_subcell_i*3 + 2 ] };

        // compute friction force
        const Vec3d relative_velocity = flow_velocity - v;
        const double relative_velocity_norm = norm(relative_velocity);
        const Vec3d friction = relative_velocity * relative_velocity_norm;
        fx += friction.x;
        fy += friction.y;
        fz += friction.z;
      }

    }
  };

  template<class GridT> struct ComputeCellParticlesTraits< ParticleFluidFrictionFunctor<GridT> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz >
    >
  class FluidFriction : public OperatorNode
  {  
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
    ADD_SLOT( bool  , ghost , INPUT, false );
    ADD_SLOT( GridCellValues, grid_cell_values , INPUT );
    ADD_SLOT( std::string   , field_name       , INPUT , REQUIRED );

    static constexpr FieldSet< field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz > compute_field_set{};
    
  public:
    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) return;

      const auto field = grid_cell_values->field(*field_name);
      const unsigned int subdiv = field.m_subdiv;
      if( field.m_components != subdiv*subdiv*subdiv*3 )
      {
        fatal_error() << "field "<< *field_name <<" must have exactly 3 compnents, but has "<< field.m_components*1.0/(subdiv*subdiv*subdiv) << std::endl;
      }
      const auto field_data = grid_cell_values->field_data(*field_name);
      
      ParticleFluidFrictionFunctor<GridT> func = { *grid , field_data.m_data_ptr , field_data.m_stride , subdiv };
      
      compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );                  
    }

  };

  template<class GridT> using FluidFrictionTmpl = FluidFriction< GridT >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "fluid_friction", make_grid_variant_operator< FluidFrictionTmpl > );
  }

}

