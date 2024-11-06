/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <functional>

#include <onika/cpp_utils.h>
#include <onika/plugin.h>

namespace onika
{

  struct UnitTestsReport
  {
    int n_passed = 0;
    int n_failed = 0;
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

#define ONIKA_TEST_ASSERT(c) if(!(c)) { std::cerr<<"Assertion '"<< #c <<"' failed"<<std::endl<<std::flush; throw std::exception(); } //

#define ONIKA_UNIT_TEST(name) \
extern "C" { extern void __onika_unit_test_##name(); } \
CONSTRUCTOR_ATTRIB inline void __onika_unit_test_register_##name()\
{ \
  std::function<void()> test_func = __onika_unit_test_##name; \
  ::onika::UnitTest::register_unit_test( #name , test_func ); \
  ::onika::plugin_db_register( "unit_test" , #name ); \
} \
inline void __onika_unit_test_##name()

