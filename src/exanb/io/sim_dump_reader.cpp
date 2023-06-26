#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_reader.h>

namespace exanb
{
  

  template<class GridT> using SimDumpReadPositions = SimDumpReader<GridT, FieldSet<field::_rx,field::_ry,field::_rz> >;
  template<class GridT> using SimDumpReadAll = SimDumpReader<GridT, typename GridT::Fields >;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "read_dump_r" , make_grid_variant_operator<SimDumpReadPositions> );
    OperatorNodeFactory::instance()->register_factory( "read_dump_all" , make_grid_variant_operator<SimDumpReadAll> );
  }

}

