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

#include <cstdint>

namespace exanb
{
  class CellArraySerializer
  {
  public:
    virtual ~CellArraySerializer() {}
    virtual size_t number_of_cells() =0;
    virtual size_t cell_stream_size(size_t cell_index) =0;
    virtual void serialize_cell(size_t cell_index, uint8_t* output_buffer) =0;
  };

  class CellArrayDeserializer
  {
  public:
    virtual ~CellArrayDeserializer() {}
    virtual void set_number_of_cells(size_t n_cells) =0;
    virtual void deserialize_cell(size_t cell_index, const uint8_t* input_buffer, size_t stream_size ) =0;
  };

  struct CellArrayStreamer
  {
    std::shared_ptr<CellArraySerializer> m_serializer;
    std::shared_ptr<CellArrayDeserializer> m_deserializer;
  };

}

