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

#include <onika/math/basic_types.h>
#include <mpi.h>
#include <string>

namespace exanb
{

  class LatticeCollection {
    using StringVector = std::vector<std::string>;
    using Vec3dVector = std::vector<Vec3d>;
    
  public:
    std::string m_structure = "UNDEFINED";
    ssize_t m_np = 0;
    StringVector m_types;
    Vec3dVector m_positions;
    Vec3d m_size;
    Mat3d m_rotmat;
  };

  // Function to generate a random double between min and max
  double random_double(double min, double max) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
  }
  
  // Function to generate a random unit vector uniformly distributed on a sphere
  Vec3d random_unit_vector() {
    double theta = random_double(0.0, 2.0 * M_PI);
    double z = random_double(-1.0, 1.0);
    double r = std::sqrt(1.0 - z * z);
    Vec3d k = {r * std::cos(theta), r * std::sin(theta), z};
    double norm = std::sqrt(k.x*k.x+k.y*k.y+k.z*k.z);
    std::cout << "Vector norm: " << norm << "\n";
    return k;
  }
  
  // Rodrigues' rotation formula to compute rotation matrix
  Mat3d rotation_matrix(const Vec3d axis, double angle) {
    double x = 0.;//axis.x;
    double y = 1.;//axis.y;
    double z = 0.;//axis.z;
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;
    
    Mat3d R = {t*x*x + c  , t*x*y - s*z, t*x*z + s*y,
               t*x*y + s*z, t*y*y + c  , t*y*z - s*x,
               t*x*z - s*y, t*y*z + s*x, t*z*z + c  };
    std::cout << "Rotation matrix = " << R << std::endl;
    return R;
  }
  
  // Main function to generate a random rotation matrix
  Mat3d random_rotation_matrix() {
    Vec3d axis = random_unit_vector();
    double angle = random_double(0.0, 2.0 * M_PI);
    return rotation_matrix(axis, angle);
  }
  
}
