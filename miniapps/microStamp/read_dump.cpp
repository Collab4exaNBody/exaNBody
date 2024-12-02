#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>
#include <exanb/core/file_utils.h>

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_reader.h>

namespace exaStamp
{
  using namespace exanb;

  using BoolVector = std::vector<bool>;

  template<
    class GridT,
    class = AssertGridHasFields< GridT, field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_id, field::_type >
    >
  class ReadDump : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( std::string , filename , INPUT );
    ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
    ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );

    ADD_SLOT( GridT       , grid     , INPUT_OUTPUT );
    ADD_SLOT( Domain      , domain   , INPUT_OUTPUT );

  public:
    inline void execute () override final
    {
      static constexpr FieldSet< field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_id, field::_type > dump_field_set={};
      std::string file_name = data_file_path( *filename );
      *physical_time = 0.0;
      *timestep = 0;
      exanb::read_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, file_name, dump_field_set );
    }
  };

  template<class GridT> using ReadDumpTmpl = ReadDump<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "read_dump" , make_grid_variant_operator<ReadDumpTmpl> );
  }

}

