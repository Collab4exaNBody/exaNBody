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
 *   SimulationStatistics  →  Python dict
 *   GridCellValues        →  GridCellValuesView (fields returned as numpy copies)
 *   ParticleFieldBundle   →  dict of numpy arrays (output of extract_particle_fields)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <onika/scg/operator_slot.h>
#include <onika/python/slot_extractor_api.h>
#include <exanb/compute/simulation_statistics.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/particle_field_bundle.h>

#include <cstring>
#include <functional>
#include <typeindex>

namespace py = pybind11;
using OSB = onika::scg::OperatorSlotBase;

// ---------------------------------------------------------------------------
// Retrieve register_slot_extractor via pyonika's Python capsule attribute.
// ---------------------------------------------------------------------------
using RegisterFn = onika::python::RegisterSlotExtractorFn;

// versioned registration function published by pyonika (ImportError if incompatible)
static RegisterFn get_register_fn()
{
  return onika::python::slot_extractor_registry();
}

// ---------------------------------------------------------------------------
// SimulationStatistics extractor
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
// GridCellValues wrapper and extractor
//
// GCVView holds a raw pointer to GridCellValues in the operator slot: keep the
// ApplicationContext alive while the view is used.  Arrays returned by field()
// and field_inner() are copies (pybind11 copies a buffer_info without base).
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
// ParticleFieldBundle extractor: one numpy array per field, shape (count,)+shape,
// in the field's native dtype.  Arrays own a copy of the data.
// ---------------------------------------------------------------------------
static void register_particle_field_bundle(RegisterFn reg)
{
  using PFB = exanb::ParticleFieldBundle;
  reg(
    std::type_index(typeid(PFB)),
    [](OSB& slot) -> py::object {
      auto* typed = static_cast<onika::scg::OperatorSlot<PFB>*>(&slot);
      if (!typed->has_value()) return py::none();
      const PFB& b = **typed;
      py::dict d;
      for (const auto& [name, col] : b.columns)
      {
        std::vector<py::ssize_t> shape { static_cast<py::ssize_t>(b.count) };
        for (auto s : col.shape) shape.push_back(static_cast<py::ssize_t>(s));
        py::array a( py::dtype("=" + col.dtype), shape );
        if (!col.data.empty()) std::memcpy(a.mutable_data(), col.data.data(), col.data.size());
        d[py::str(name)] = a;
      }
      return d;
    }
  );
}

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_exanb_data, m)
{
  m.doc() = "exaNBody-specific slot extractors for pyonika's slot_as_array registry";

  // Wrapper class exposing GridCellValues fields as numpy arrays (copies).
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
    // field(name) — view of ALL cells including ghost layers.
    // subdiv==1, ncomps==1 → shape (n_cells,)
    // subdiv==1, ncomps> 1 → shape (n_cells, ncomps)
    // subdiv> 1, ncomps==1 → shape (n_cells, S, S, S)
    // subdiv> 1, ncomps> 1 → shape (n_cells, S, S, S, ncomps)
    .def("field", [](const GCVView& v, const std::string& name) -> py::object {
      if (!v.ptr->has_field(name)) return py::none();
      const auto& f  = v.ptr->field(name);
      size_t n_cells = v.ptr->number_of_cells();
      size_t tot     = v.ptr->components();
      size_t S       = f.m_subdiv;
      size_t ncomps  = f.m_components / (S * S * S);
      const double* base = v.ptr->data().data() + f.m_offset;

      py::ssize_t s_cell = static_cast<py::ssize_t>(tot       * sizeof(double));
      py::ssize_t s_ci   = static_cast<py::ssize_t>(S*S*ncomps * sizeof(double));
      py::ssize_t s_cj   = static_cast<py::ssize_t>(S  *ncomps * sizeof(double));
      py::ssize_t s_ck   = static_cast<py::ssize_t>(     ncomps * sizeof(double));
      py::ssize_t s_comp = static_cast<py::ssize_t>(sizeof(double));

      if (S == 1 && ncomps == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 1,
          { static_cast<py::ssize_t>(n_cells) }, { s_cell }));
      if (S == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 2,
          { static_cast<py::ssize_t>(n_cells), static_cast<py::ssize_t>(ncomps) },
          { s_cell, s_comp }));
      if (ncomps == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 4,
          { static_cast<py::ssize_t>(n_cells), static_cast<py::ssize_t>(S),
            static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(S) },
          { s_cell, s_ci, s_cj, s_ck }));
      return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
        py::format_descriptor<double>::format(), 5,
        { static_cast<py::ssize_t>(n_cells), static_cast<py::ssize_t>(S),
          static_cast<py::ssize_t>(S), static_cast<py::ssize_t>(S),
          static_cast<py::ssize_t>(ncomps) },
        { s_cell, s_ci, s_cj, s_ck, s_comp }));
    })
    // field_inner(name) — ghost-stripped view of inner domain cells.
    // subdiv==1, ncomps==1 → shape (inx, iny, inz)
    // subdiv==1, ncomps> 1 → shape (inx, iny, inz, ncomps)
    // subdiv> 1, ncomps==1 → shape (inx, iny, inz, S, S, S)
    // subdiv> 1, ncomps> 1 → shape (inx, iny, inz, S, S, S, ncomps)
    // Cell and sub-cell strides skip the ghost border.
    .def("field_inner", [](const GCVView& v, const std::string& name) -> py::object {
      if (!v.ptr->has_field(name)) return py::none();
      const auto& f = v.ptr->field(name);
      const auto& d = v.ptr->grid_dims();
      size_t gl     = v.ptr->ghost_layers();
      size_t tot    = v.ptr->components();
      size_t nx = static_cast<size_t>(d.i), ny = static_cast<size_t>(d.j), nz = static_cast<size_t>(d.k);
      size_t inx = nx-2*gl, iny = ny-2*gl, inz = nz-2*gl;
      size_t S      = f.m_subdiv;
      size_t ncomps = f.m_components / (S * S * S);

      size_t inner_offset = (gl*ny*nz + gl*nz + gl) * tot + f.m_offset;
      const double* base  = v.ptr->data().data() + inner_offset;

      py::ssize_t si   = static_cast<py::ssize_t>(ny*nz*tot   * sizeof(double));
      py::ssize_t sj   = static_cast<py::ssize_t>(nz *tot     * sizeof(double));
      py::ssize_t sk   = static_cast<py::ssize_t>(    tot     * sizeof(double));
      py::ssize_t s_ci = static_cast<py::ssize_t>(S*S*ncomps  * sizeof(double));
      py::ssize_t s_cj = static_cast<py::ssize_t>(S  *ncomps  * sizeof(double));
      py::ssize_t s_ck = static_cast<py::ssize_t>(    ncomps  * sizeof(double));
      py::ssize_t s_c  = static_cast<py::ssize_t>(sizeof(double));

      auto I = [](size_t x) { return static_cast<py::ssize_t>(x); };

      if (S == 1 && ncomps == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 3,
          {I(inx),I(iny),I(inz)}, {si,sj,sk}));
      if (S == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 4,
          {I(inx),I(iny),I(inz),I(ncomps)}, {si,sj,sk,s_c}));
      if (ncomps == 1)
        return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
          py::format_descriptor<double>::format(), 6,
          {I(inx),I(iny),I(inz),I(S),I(S),I(S)}, {si,sj,sk,s_ci,s_cj,s_ck}));
      return py::array_t<double>(py::buffer_info(const_cast<double*>(base), sizeof(double),
        py::format_descriptor<double>::format(), 7,
        {I(inx),I(iny),I(inz),I(S),I(S),I(S),I(ncomps)}, {si,sj,sk,s_ci,s_cj,s_ck,s_c}));
    });

  RegisterFn reg = get_register_fn();
  register_simulation_statistics(reg);
  register_grid_cell_values(reg);
  register_particle_field_bundle(reg);
}
