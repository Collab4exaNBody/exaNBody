// this is a test, it always needs enabled asserts
#ifndef NDEBUG
#define NDEBUG 1
#endif

#include <exanb/core/grid.h>
#include <exanb/fields.h>

#include <iostream>
#include <cstdlib>

struct A
{
  int memory_bytes(int y);
};

int main(int argc,char*argv[])
{
  using std::cout;
  using std::endl;
  

  auto grid = make_grid( field::ep, field::ax, field::ay, field::az, field::vx, field::vy, field::vz, field::id, field::type );
  std::cout<<"grid.memory_bytes() = "<<grid.memory_bytes()<<std::endl;

  static_assert( onika::memory::has_memory_bytes_method_v<decltype(grid)> , "a valid memory_bytes method is needed for proper memory usage accounting" );
  A a;
  static_assert( ! onika::memory::has_memory_bytes_method_v<decltype(a)> , "shouldn't be detected as having a valid memorty_bytes method");

  return 0;
}


