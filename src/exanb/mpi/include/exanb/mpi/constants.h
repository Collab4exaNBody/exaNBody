#pragma once
#include <mpi.h>

namespace exanb
{

/****** generation code ******
cat > regen.py <<- EOF
import sys
for s in sys.stdin.readlines():
  c = s[:-1]
  print("__attribute__((always_inline)) static inline auto __xstamp_%s() { return %s; }\n#undef %s\n#define %s ::exanb::__xstamp_%s()" %(c,c,c,c,c) )
EOF
python3 regen.py <<- EOF
MPI_CHAR
MPI_INT
MPI_LONG
MPI_UNSIGNED_LONG
MPI_DOUBLE
MPI_MAX
MPI_MIN
MPI_SUM
MPI_REQUEST_NULL
EOF
*****************************/

__attribute__((always_inline)) static inline auto __xstamp_MPI_CHAR() { return MPI_CHAR; }
#undef MPI_CHAR
#define MPI_CHAR ::exanb::__xstamp_MPI_CHAR()
__attribute__((always_inline)) static inline auto __xstamp_MPI_INT() { return MPI_INT; }
#undef MPI_INT
#define MPI_INT ::exanb::__xstamp_MPI_INT()
__attribute__((always_inline)) static inline auto __xstamp_MPI_LONG() { return MPI_LONG; }
#undef MPI_LONG
#define MPI_LONG ::exanb::__xstamp_MPI_LONG()
__attribute__((always_inline)) static inline auto __xstamp_MPI_UNSIGNED_LONG() { return MPI_UNSIGNED_LONG; }
#undef MPI_UNSIGNED_LONG
#define MPI_UNSIGNED_LONG ::exanb::__xstamp_MPI_UNSIGNED_LONG()
__attribute__((always_inline)) static inline auto __xstamp_MPI_DOUBLE() { return MPI_DOUBLE; }
#undef MPI_DOUBLE
#define MPI_DOUBLE ::exanb::__xstamp_MPI_DOUBLE()
__attribute__((always_inline)) static inline auto __xstamp_MPI_MAX() { return MPI_MAX; }
#undef MPI_MAX
#define MPI_MAX ::exanb::__xstamp_MPI_MAX()
__attribute__((always_inline)) static inline auto __xstamp_MPI_MIN() { return MPI_MIN; }
#undef MPI_MIN
#define MPI_MIN ::exanb::__xstamp_MPI_MIN()
__attribute__((always_inline)) static inline auto __xstamp_MPI_SUM() { return MPI_SUM; }
#undef MPI_SUM
#define MPI_SUM ::exanb::__xstamp_MPI_SUM()
__attribute__((always_inline)) static inline auto __xstamp_MPI_REQUEST_NULL() { return MPI_REQUEST_NULL; }
#undef MPI_REQUEST_NULL
#define MPI_REQUEST_NULL ::exanb::__xstamp_MPI_REQUEST_NULL()

}

