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

#include <onika/app/api.h>

int main(int argc,char*argv[])
{
  // ============= run simulation ================
  auto app = onika::app::init(argc,argv);
  if( app->get_error_code() >= 0 ) return app->get_error_code();
  
  // mandatory if multiple invocations of onika::app::run( xxx ); are required
  app->set_multiple_run( true );

  // set some input value by copy
  app->node("sim.loop")->in_slot("dt")->copy_input_value( 1.5e-6 );
  // ... or by referencing main application's data
  // double dt=1.5e-6; app->node("sim.loop")->in_slot("dt")->set_value_pointer( &dt );
  
  // run full simulation graph
  onika::app::run( app );
  // ... or a portion of it
  // onika::app::run( app->node("sim.loop.core") );
  
  // read back some output from simulation graph
  const double * output_h = app->node("sim.loop")->out_slot("h")->output_value_pointer<double>();
  std::cout << "h value after iteration = " << *output_h << std::endl;
  
  // finalize simulation graph
  onika::app::end( app );
  
  return 0;
}

