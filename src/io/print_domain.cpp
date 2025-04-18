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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <string>

namespace exanb
{

  struct PrintDomain : public OperatorNode
  {
    ADD_SLOT( Domain , domain , INPUT );

    inline void execute() override final
    {
      const char* mat_property = "";
      if( is_diagonal(domain->xform()) )
      {
        if( is_identity(domain->xform()) ) mat_property = " (identity)";
        else mat_property = " (diagonal)";
      }

      const char* sep="";

      lout << std::defaultfloat
           << "======= Simulation Domain ======="<< std::endl
           << "bounds    = " << domain->bounds() <<std::endl
           << "dom size  = " << (domain->bounds().bmax - domain->bounds().bmin ) <<std::endl
           << "grid      = " << domain->grid_dimension() << std::endl
           << "cell size = " << domain->cell_size() << std::endl
           << "grid size = " << domain->grid_dimension() * domain->cell_size() << std::endl
           << "periodic  = " << std::boolalpha << domain->periodic_boundary_x()<<" , "<<domain->periodic_boundary_y()<<" , "<<domain->periodic_boundary_z() << std::endl
           << "mirror    = " ;
      if( domain->mirror_x_min() ) { lout << sep << "X-"; sep=" , "; }
      if( domain->mirror_x_max() ) { lout << sep << "X+"; sep=" , "; }
      if( domain->mirror_y_min() ) { lout << sep << "Y-"; sep=" , "; }
      if( domain->mirror_y_max() ) { lout << sep << "Y+"; sep=" , "; }
      if( domain->mirror_z_min() ) { lout << sep << "Z-"; sep=" , "; }
      if( domain->mirror_z_max() ) { lout << sep << "Z+"; sep=" , "; }
      lout << std::endl
           << "xform     = " << domain->xform() << mat_property << std::endl
           << "inv_xform = " << domain->inv_xform() << std::endl
           << "scale     = " << domain->xform_min_scale()<< " / " <<domain->xform_max_scale() << std::endl;
      if( is_diagonal(domain->xform()) && ! is_identity(domain->xform()) )
      {
        auto dom_size = domain->xform() * (domain->bounds().bmax - domain->bounds().bmin);
        auto cell_size = domain->xform() * Vec3d { domain->cell_size() , domain->cell_size() , domain->cell_size() };
        lout << "phys size = " << dom_size << " , cell size = "<< cell_size <<std::endl;
      }
      lout << "================================="<< std::endl << std::endl;      
    }

  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(print_domain)
  {
    OperatorNodeFactory::instance()->register_factory( "print_domain", make_simple_operator<PrintDomain> );
  }

}

