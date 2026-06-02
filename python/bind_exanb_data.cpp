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

/*
 * _exanb_data — pybind11 extension that registers exaNBody-specific slot
 * extractors into pyonika's slot_as_array registry.
 *
 * Importing this module (done automatically by pyexanbody/__init__.py) adds
 * support for types that pyonika itself cannot know about:
 *
 *   Tier 1 — SimulationStatistics  →  Python dict
 *   Tier 2 — GridCellValues        →  GridCellValuesView (zero-copy numpy fields)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <onika/scg/operator_slot.h>
#include <exanb/compute/simulation_statistics.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <functional>
#include <typeindex>

namespace py = pybind11;
using OSB = onika::scg::OperatorSlotBase;

// ---------------------------------------------------------------------------
// Retrieve register_slot_extractor via pyonika's Python capsule attribute.
// ---------------------------------------------------------------------------
using SlotExtractorFn = std::function<py::object(OSB&)>;
using RegisterFn = void(*)(std::type_index, SlotExtractorFn);

static RegisterFn get_register_fn()
{
  py::module_ pyonika = py::module_::import("pyonika");
  py::capsule cap = pyonika.attr("_register_slot_extractor_fn").cast<py::capsule>();
  return reinterpret_cast<RegisterFn>(cap.get_pointer());
}

// ---------------------------------------------------------------------------
// Tier 1: SimulationStatistics extractor
// ---------------------------------------------------------------------------
static void register_simulation_statistics(RegisterFn reg)
{
  using SS = exanb::SimulationStatistics;
  reg(
    std::type_index(typeid(SS)),
    [](OSB& slot) -> py::object {
      auto* typed = static_cast<onika::scg::OperatorSlot<SS>*>(&slot);
      if (!typed->has_value()) return py::none();
      const SS& s = **typed;
      py::dict d;
      d["kinetic_energy"]  = s.m_kinetic_energy;
      d["particle_count"]  = static_cast<unsigned long long>(s.m_particle_count);
      d["min_vel"]         = s.m_min_vel;
      d["max_vel"]         = s.m_max_vel;
      d["min_acc"]         = s.m_min_acc;
      d["max_acc"]         = s.m_max_acc;
      return d;
    }
  );
}

// ---------------------------------------------------------------------------
// Tier 2: GridCellValues wrapper and extractor
//
// GCVView holds a raw pointer to GridCellValues in the operator slot.
// The caller must keep the ApplicationContext alive for as long as any
// numpy array produced by GCVView.field() is in use.
// ---------------------------------------------------------------------------
struct GCVView
{
  const exanb::GridCellValues* ptr;
};

static void register_grid_cell_values(RegisterFn reg)
{
  using GCV = exanb::GridCellValues;
  reg(
    std::type_index(typeid(GCV)),
    [](OSB& slot) -> py::object {
      auto* typed = static_cast<onika::scg::OperatorSlot<GCV>*>(&slot);
      if (!typed->has_value()) return py::none();
      const GCV& gcv = **typed;
      if (gcv.empty()) return py::none();
      return py::cast(GCVView{&gcv});
    }
  );
}

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_exanb_data, m)
{
  m.doc() = "exaNBody-specific slot extractors for pyonika's slot_as_array registry";

  // Tier 2 wrapper class — exposes GridCellValues fields as zero-copy numpy arrays.
  py::class_<GCVView>(m, "GridCellValuesView")
    .def_property_readonly("field_names", [](const GCVView& v) {
      py::list names;
      for (const auto& [name, _] : v.ptr->fields())
        names.append(name);
      return names;
    })
    .def_property_readonly("shape", [](const GCVView& v) {
      const auto& d = v.ptr->grid_dims();
      return py::make_tuple(d.i, d.j, d.k);
    })
    .def_property_readonly("n_cells", [](const GCVView& v) {
      return v.ptr->number_of_cells();
    })
    .def_property_readonly("ghost_layers", [](const GCVView& v) {
      return v.ptr->ghost_layers();
    })
    .def("has_field", [](const GCVView& v, const std::string& name) {
      return v.ptr->has_field(name);
    })
    // field(name) — flat 1-D (or 2-D) view of ALL cells including ghost layers.
    .def("field", [](const GCVView& v, const std::string& name) -> py::object {
      if (!v.ptr->has_field(name)) return py::none();
      const auto& f    = v.ptr->field(name);
      size_t n_cells   = v.ptr->number_of_cells();
      size_t tot       = v.ptr->components();
      const double* base = v.ptr->data().data() + f.m_offset;

      py::ssize_t cell_stride = static_cast<py::ssize_t>(tot * sizeof(double));
      py::ssize_t comp_stride = static_cast<py::ssize_t>(sizeof(double));

      if (f.m_components == 1) {
        return py::array_t<double>(py::buffer_info(
          const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(),
          1,
          { static_cast<py::ssize_t>(n_cells) },
          { cell_stride }
        ));
      } else {
        return py::array_t<double>(py::buffer_info(
          const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(),
          2,
          { static_cast<py::ssize_t>(n_cells), static_cast<py::ssize_t>(f.m_components) },
          { cell_stride, comp_stride }
        ));
      }
    })
    // field_inner(name) — 3-D (or 4-D) strided view with ghost layers stripped.
    // Returns shape (inx, iny, inz) for scalar fields,
    //               (inx, iny, inz, n_components) for multi-component fields,
    // where inx/iny/inz = grid_dims - 2*ghost_layers on each axis.
    // Zero-copy: the strides span over the ghost border without touching it.
    .def("field_inner", [](const GCVView& v, const std::string& name) -> py::object {
      if (!v.ptr->has_field(name)) return py::none();
      const auto& f  = v.ptr->field(name);
      const auto& d  = v.ptr->grid_dims();
      size_t gl      = v.ptr->ghost_layers();
      size_t tot     = v.ptr->components();
      size_t nx = static_cast<size_t>(d.i);
      size_t ny = static_cast<size_t>(d.j);
      size_t nz = static_cast<size_t>(d.k);
      size_t inx = nx - 2*gl, iny = ny - 2*gl, inz = nz - 2*gl;

      // Pointer to inner cell (gl, gl, gl).
      size_t inner_offset = (gl*ny*nz + gl*nz + gl) * tot + f.m_offset;
      const double* base  = v.ptr->data().data() + inner_offset;

      // Byte strides for a cell-major (i,j,k) layout.
      py::ssize_t si = static_cast<py::ssize_t>(ny * nz * tot * sizeof(double));
      py::ssize_t sj = static_cast<py::ssize_t>(nz * tot * sizeof(double));
      py::ssize_t sk = static_cast<py::ssize_t>(tot * sizeof(double));
      py::ssize_t sc = static_cast<py::ssize_t>(sizeof(double));

      if (f.m_components == 1) {
        return py::array_t<double>(py::buffer_info(
          const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(),
          3,
          { static_cast<py::ssize_t>(inx), static_cast<py::ssize_t>(iny), static_cast<py::ssize_t>(inz) },
          { si, sj, sk }
        ));
      } else {
        return py::array_t<double>(py::buffer_info(
          const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(),
          4,
          { static_cast<py::ssize_t>(inx), static_cast<py::ssize_t>(iny),
            static_cast<py::ssize_t>(inz), static_cast<py::ssize_t>(f.m_components) },
          { si, sj, sk, sc }
        ));
      }
    });

  RegisterFn reg = get_register_fn();
  register_simulation_statistics(reg);
  register_grid_cell_values(reg);
}
