#pragma once

template<typename... field_ids>
struct DummyGrid
{
  static int dummy_grid_function_a();
  int dummy_grid_function_b();
  int m = 0;
};

