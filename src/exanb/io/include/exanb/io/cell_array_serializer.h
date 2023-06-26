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

