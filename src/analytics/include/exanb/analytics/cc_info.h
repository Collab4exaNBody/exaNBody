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

#include <onika/math/basic_types.h>
#include <vector>

namespace exanb
{
  struct ConnectedComponentInfo
  {
    ssize_t m_rank = -1;
    ssize_t m_cell_count = 0;
    double m_label = -1.0;
    Vec3d m_center = {0.,0.,0.};
    Mat3d m_gyration = { 0.,0.,0., 0.,0.,0., 0.,0.,0. };
  };

  struct ConnectedComponentTable
  {
    std::vector< ConnectedComponentInfo > m_table;
    size_t m_global_label_count = 0;
    size_t m_local_label_start = 0;

    inline auto begin() { return m_table.begin(); }
    inline auto begin() const { return m_table.begin(); }
    inline auto end() { return m_table.end(); }
    inline auto end() const { return m_table.end(); }
    inline ConnectedComponentInfo & at(size_t i) { return m_table.at(i); }
    inline const ConnectedComponentInfo & at(size_t i) const { return m_table.at(i); }
    inline size_t size() const { return m_table.size(); }
    inline void resize(size_t n, const ConnectedComponentInfo & def = ConnectedComponentInfo{} ) { return m_table.resize(n,def); }
    inline void assign(size_t n, const ConnectedComponentInfo & def ) { return m_table.assign(n,def); }
  };

}

