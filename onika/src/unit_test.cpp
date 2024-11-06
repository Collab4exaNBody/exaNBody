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
#include <onika/test/unit_test.h>
#include <onika/log.h>
#include <iostream>

namespace onika
{
  UnitTest* UnitTest::s_unit_tests_list = nullptr;

  void UnitTest::register_unit_test(const std::string& name , std::function<void()> F )
  {
    UnitTest* ut = UnitTest::s_unit_tests_list;    
    while( ut != nullptr )
    {
      if( name == ut->m_name )
      {
        ldbg << "unit test "<<name<<" already registered, skipping this registration"<<std::endl;
        return;
      }
      ut=ut->m_next;
    }
    ut = new UnitTest { name , F , s_unit_tests_list };
    s_unit_tests_list = ut;
    if( ! quiet_plugin_register() )
    {
      lout<<"  unit_test   "<< name << std::endl;
    }
  }
  
  UnitTestsReport UnitTest::run_unit_tests()
  {
    int n_passed = 0;
    int n_failed = 0;
    
    UnitTest* ut = UnitTest::s_unit_tests_list;    
    int n_tests = 0;
    while( ut != nullptr ) { ++n_tests; ut=ut->m_next; }
    
    ut = UnitTest::s_unit_tests_list;
    while( ut != nullptr )
    {
      lout << "Unit test "<<(n_passed+n_failed+1)<<"/"<<n_tests<<" : "<< ut->m_name << " ... " << std::flush ;
      bool test_passed = true;
      try
      {
        ut->m_test ();
      }
      catch(const std::exception& e)
      {
        test_passed = false;
      }
      if(test_passed)
      {
        lout << "Ok" << std::endl;
        ++ n_passed;
      }
      else
      {
        lout << "FAILED" << std::endl;
        ++ n_failed;
      }
      ut = ut->m_next;
    }
    return { n_passed , n_failed };
  }
  
}
