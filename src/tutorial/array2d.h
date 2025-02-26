#pragma once

#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>

// a specific namespace for our application space
namespace tutorial
{
  using namespace exanb;
  
  struct Array2D
  {
    onika::memory::CudaMMVector<double> m_data;
    size_t m_rows = 0;              // number of rows in our array
    size_t m_columns = 0;           // number of columns in our array
    inline void resize(size_t r, size_t c) { m_rows=r; m_columns=c; m_data.resize(r*c); }
    ONIKA_HOST_DEVICE_FUNC inline double const * data() const { return onika::cuda::vector_data( m_data ); }
    ONIKA_HOST_DEVICE_FUNC inline double * data() { return onika::cuda::vector_data( m_data ); }
    ONIKA_HOST_DEVICE_FUNC inline double const * operator [] (size_t row) const { return data() + ( m_columns * row ); }
    ONIKA_HOST_DEVICE_FUNC inline double * operator [] (size_t row) { return data() + ( m_columns * row ); }
    ONIKA_HOST_DEVICE_FUNC inline size_t columns() const { return m_columns; }
    ONIKA_HOST_DEVICE_FUNC inline size_t rows() const { return m_rows; }
  };  

  struct Array2DReference
  {
    double * const m_data_ptr = nullptr; // our 2D array
    size_t const m_rows = 0;              // number of rows in our array
    size_t const m_columns = 0;           // number of columns in our array
    Array2DReference(Array2DReference && a) = default;
    Array2DReference(const Array2DReference & a) = default;
    Array2DReference() = default;
    ONIKA_HOST_DEVICE_FUNC inline Array2DReference(Array2D& a) : m_data_ptr(a.data()) , m_rows(a.rows()) , m_columns(a.columns()) {}
    ONIKA_HOST_DEVICE_FUNC inline double * operator [] (size_t row) const { return m_data_ptr + ( m_columns * row ); }
    ONIKA_HOST_DEVICE_FUNC inline size_t columns() const { return m_columns; }
    ONIKA_HOST_DEVICE_FUNC inline size_t rows() const { return m_rows; }
  };  
}

