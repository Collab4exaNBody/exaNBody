#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <functional>

#include <exanb/core/cpp_utils.h>
#include <exanb/core/plugin.h>

namespace exanb
{

  struct UnitTestsReport
  {
    int n_passed;
    int n_failed;
  };

  struct UnitTest
  {
    const std::string m_name;
    std::function<void()> m_test = nullptr;
    UnitTest* m_next = nullptr;

    static UnitTest* s_unit_tests_list;
    static UnitTestsReport run_unit_tests();
    static void register_unit_test(const std::string& name , std::function<void()> F );
  };
  
}

#define XSTAMP_TEST_ASSERT(c) if(!(c)) { std::cerr<<"Assertion '"<< #c <<"' failed"<<std::endl<<std::flush; throw std::exception(); } //

#define XSTAMP_UNIT_TEST(name) \
extern "C" { extern void __exanb_unit_test_##name(); } \
CONSTRUCTOR_ATTRIB inline void __exanb_unit_test_register_##name()\
{ \
  std::function<void()> test_func = __exanb_unit_test_##name; \
  ::exanb::UnitTest::register_unit_test( #name , test_func ); \
  ::exanb::plugin_db_register( "unit_test" , #name ); \
} \
inline void __exanb_unit_test_##name()

