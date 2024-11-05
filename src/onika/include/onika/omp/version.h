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

#include <utility>
#include <sstream>

#include <omp.h>

namespace onika
{

  namespace omp
  {
  
    inline double get_version()
    {
      double version = 1.0;
      std::pair<unsigned long,double> version_dates [] = { {200505,2.5},{200805,3.0},{201107,3.1},{201307,4.0},{201511,4.5},{201811,5.0},{202011,5.1} };
      for(int i=0;i<7;i++)
      {
        if( version_dates[i].first < _OPENMP ) version = version_dates[i].second;
      }
      return version;
    }
    
    inline std::string get_version_string()
    {
      double v = get_version();
      std::ostringstream oss;
      oss<< static_cast<int>(std::floor(v)) << '.' << ( static_cast<int>(std::floor(v*10))%10 );
      return oss.str();
    }

  }

}
