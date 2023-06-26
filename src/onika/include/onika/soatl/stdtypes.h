#pragma once

#include <cstdint>

namespace onika { namespace soatl {

# ifdef SOATL_SIZE_TYPE_32BITS
  using array_size_t = uint32_t;
# else
  using array_size_t = uint64_t;
# endif

} }

