#pragma once

namespace exanb
{

  enum SlotDirection
  {
    INPUT=1,
    OUTPUT=2,
    INPUT_OUTPUT=3
  };

  static inline constexpr SlotDirection PRIVATE = INPUT_OUTPUT;

  const char* slot_dir_str(SlotDirection d);

}

