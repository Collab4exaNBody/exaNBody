#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/fields.h>
#include <exanb/core/parallel_random.h>

#include <exanb/grid_cell_particles/particle_localized_filter.h>

namespace exanb
{

  template<class RNG, class CellT, class ParticleFilterFuncT, class IdField, class... field_ids>
  static inline void apply_gaussian_noise( CellT& cell, RNG& re, double sigma,
            const ParticleFilterFuncT& particle_filter,
            IdField ID,
            FieldSet<field_ids...> )
  {
    static constexpr bool has_id_field = CellT::has_field( ID );
    std::normal_distribution<double> gaussian(0.0,sigma);
    const size_t n = cell.size();
    for(size_t i=0;i<n;i++)
    {
      Vec3d r = { cell[field::rx][i] , cell[field::ry][i] , cell[field::rz][i] };
      uint64_t id = 0;
      if constexpr ( has_id_field ) { id = cell[ID][i]; }
      if( particle_filter(r,id) )
      {
        ( ... , ( cell[onika::soatl::FieldId<field_ids>()] [i] += gaussian(re) ) );
      }
    }
  }

  template<
    class GridT,
    class IdField,
    class FieldSetT,
    class = AssertGridContainFieldSet< GridT, FieldSetT >
    >
  class GaussianNoise : public OperatorNode
  {
    ADD_SLOT( GridT  , grid    , INPUT_OUTPUT );
    ADD_SLOT( Domain , domain  , INPUT );

    ADD_SLOT( double , sigma   , INPUT , 1.0 );
    ADD_SLOT( double , dt      , INPUT , 1.0 );
    ADD_SLOT( bool   , ghost   , INPUT , false );

    // optionaly limit noise to a geometric region
    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT_OUTPUT , OPTIONAL );

    // optionaly limit lattice generation to places where some mask has some value
    ADD_SLOT( GridCellValues , grid_cell_values    , INPUT , OPTIONAL );
    ADD_SLOT( std::string    , grid_cell_mask_name , INPUT , OPTIONAL );
    ADD_SLOT( double         , grid_cell_mask_value, INPUT , OPTIONAL );

    static constexpr onika::soatl::FieldId<IdField> field_id = {};
    static constexpr FieldSetT field_set = {};

  public:

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute ()  override final
    {
      PartcileLocalizedFilter<GridT,LinearXForm> particle_filter = { *grid, { domain->xform() } };
      particle_filter.initialize_from_optional_parameters( particle_regions.get_pointer(),
                                                           region.get_pointer(),
                                                           grid_cell_values.get_pointer(), 
                                                           grid_cell_mask_name.get_pointer(), 
                                                           grid_cell_mask_value.get_pointer() );

      double sigma_dt = (*sigma) * std::sqrt( *dt );
      auto cells = grid->cells();
      IJK dims = grid->dimension();
      ssize_t gl = 0; 
      if( ! *ghost ) { gl = grid->ghost_layers(); }
      IJK gstart { gl, gl, gl };
      IJK gend = dims - IJK{ gl, gl, gl };
      IJK gdims = gend - gstart;

#     pragma omp parallel
      {
        auto& re = rand::random_engine();
        GRID_OMP_FOR_BEGIN(gdims,_,loc, schedule(dynamic) )
        {
          size_t i = grid_ijk_to_index( dims , loc + gstart );
          apply_gaussian_noise( cells[i], re, sigma_dt, particle_filter, field_id, field_set );
        }
        GRID_OMP_FOR_END
      }
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
Apply a white gaussian noise to selected fields.
Note: processes particles in ghost layers if and only if ghost input is true
)EOF";
    }

  };

}

