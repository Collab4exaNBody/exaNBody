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

#include <cmath>
#include <onika/cuda/cuda.h>

#include <yaml-cpp/yaml.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_yaml.h>
#include <onika/physics/units.h>
#include <onika/yaml/yaml_utils.h>
#include <iomanip>

namespace exanb
{
  struct Mat4d
  {
    /* M [ line ] [ column ] */
    double m[4][4] = { {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} };
  };

  static inline constexpr Mat4d Mat4d_identity = { { {1.,0.,0.,0.} , {0.,1.,0.,0.} , {0.,0.,1.,0.} , {0.,0.,0.,1.} } };
  static inline constexpr Mat4d Mat4d_zero = { { {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} , {0.,0.,0.,0.} } };

  // pretty printing
  inline std::ostream& operator << (std::ostream& out, const Mat4d& M)
  {
    out << "[";
    for(unsigned int i=0;i<4;i++)
    {
      out << ( (i==0) ? " [" : ", [" );
      for(unsigned int j=0;j<4;j++) out << (j==0? " " : ", " ) <<std::setprecision(3) << M.m[i][j];
      out << " ]";
    }
    out << " ]";
    return out;
  }

  ONIKA_HOST_DEVICE_FUNC inline bool is_zero( const Mat4d & M )
  {
    for(unsigned int i=0;i<4;i++)
    for(unsigned int j=0;j<4;j++)
    {
      if( M.m[i][j] != 0.0 ) return false;
    }
    return true;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d transpose( const Mat4d & M )
  {
    Mat4d T;
    for(unsigned int i=0;i<4;i++)
    for(unsigned int j=0;j<4;j++)
    {
      T.m[i][j] = M.m[j][i];
    }
    return T;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d make_mat4d( const Mat3d & m_in )
  {
    Mat4d M;
    M.m[0][0] = m_in.m11;
    M.m[1][0] = m_in.m21;
    M.m[2][0] = m_in.m31;
    M.m[3][0] = 0.0;
    M.m[0][1] = m_in.m12;
    M.m[1][1] = m_in.m22;
    M.m[2][1] = m_in.m32;
    M.m[3][1] = 0.0;
    M.m[0][2] = m_in.m13;
    M.m[1][2] = m_in.m23;
    M.m[2][2] = m_in.m33;
    M.m[3][2] = 0.0;      
    M.m[0][3] = 0.0;
    M.m[1][3] = 0.0;
    M.m[2][3] = 0.0;
    M.m[3][3] = 1.0;
    return M;
  }

  struct Vec4d
  {
    double v[4] = { 0.0 , 0.0 , 0.0 , 1.0 };
  };

  // pretty printing
  inline std::ostream& operator << (std::ostream& out, const Vec4d& v)
  {
    out << "(" << v.v[0] <<","<<v.v[1]<<","<<v.v[2]<<","<<v.v[3]<<")";
    return out;
  }


  ONIKA_HOST_DEVICE_FUNC inline Vec4d make_vec4d(const Vec3d& r) { return {{r.x,r.y,r.z,1.0}}; }
  ONIKA_HOST_DEVICE_FUNC inline Vec3d make_vec3d(const Vec4d& r) { return {r.v[0]/r.v[3],r.v[1]/r.v[3],r.v[2]/r.v[3]}; }

  // V is understood as a column vector
  // result is a column vector
  ONIKA_HOST_DEVICE_FUNC inline Vec4d operator * ( const Mat4d& M, const Vec4d& V)
  {
    Vec4d r = { { 0.0, 0.0, 0.0, 0.0 } };
    for(unsigned int i=0;i<4;i++)
    for(unsigned int j=0;j<4;j++)
    {
      r.v[i] += M.m[i][j] * V.v[j];
    }
    return r;
  }

  // V is understood as a column vector
  // result is a line vector
  ONIKA_HOST_DEVICE_FUNC inline Vec4d operator * ( const Vec4d& V , const Mat4d& M )
  {
    Vec4d r = { { 0.0, 0.0, 0.0, 0.0 } };
    for(unsigned int i=0;i<4;i++)
    for(unsigned int j=0;j<4;j++)
    {
      r.v[j] += V.v[i] * M.m[i][j];
    }
    return r;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d operator * ( const Mat4d& a, const Mat4d& b)
  {
    Mat4d R = Mat4d_zero;    
    for(unsigned int i=0;i<4;i++)
    for(unsigned int j=0;j<4;j++)
    for(unsigned int k=0;k<4;k++)
    {
      R.m[i][j] += a.m[i][k] * b.m[k][j];
    }
    return R;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d translation_mat4d(const Vec3d& T)
  {
    Mat4d M = Mat4d_identity;
    M.m[0][3] = T.x;
    M.m[1][3] = T.y;
    M.m[2][3] = T.z;
    return M;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d scaling_mat4d(const Vec3d& S)
  {
    Mat4d M = Mat4d_identity;
    M.m[0][0] = S.x;
    M.m[1][1] = S.y;
    M.m[2][2] = S.z;
    return M;
  }

  /* elipsoid */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d sphere_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 1.;
    M.m[1][1] = 1.;
    M.m[2][2] = 1.;
    M.m[3][3] = -1.;
    return M;
  }

  /* X-aligned unit cone */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d x_cone_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = -1.;
    M.m[1][1] = 1.;
    M.m[2][2] = 1.;
    M.m[3][3] = 0;
    return M;
  }

  /* Y-aligned unit cone */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d y_cone_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 1.;
    M.m[1][1] = -1.;
    M.m[2][2] = 1.;
    M.m[3][3] = 0;
    return M;
  }

  /* X-aligned unit cylinder */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d x_cylinder_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 0;
    M.m[1][1] = 1.;
    M.m[2][2] = 1.;
    M.m[3][3] = -1;
    return M;
  }

  /* X-aligned unit cylinder */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d y_cylinder_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 1.;
    M.m[1][1] = 0.;
    M.m[2][2] = 1.;
    M.m[3][3] = -1;
    return M;
  }

  /* X-aligned unit cylinder */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d z_cylinder_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 1.;
    M.m[1][1] = 1.;
    M.m[2][2] = 0.;
    M.m[3][3] = -1;
    return M;
  }

  /* plane equation as a quadric */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d plane_quadric(const Plane3d& P)
  {
    // vector N is the plane's outward normal vector (i.e A,B and C coefficients)
    // of the generic plane equation Ax + By + Cz + D = 0
    Mat4d M = Mat4d_zero;
    M.m[0][3] = 0.5 * P.N.x;
    M.m[1][3] = 0.5 * P.N.y;
    M.m[2][3] = 0.5 * P.N.z;
    M.m[3][0] = 0.5 * P.N.x;
    M.m[3][1] = 0.5 * P.N.y;
    M.m[3][2] = 0.5 * P.N.z;
    M.m[3][3] = P.D;
    return M;
  }

  /* Z-aligned unit cone */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d z_cone_quadric()
  {
    Mat4d M = Mat4d_zero;
    M.m[0][0] = 1.;
    M.m[1][1] = 1.;
    M.m[2][2] = -1.;
    M.m[3][3] = 0;
    return M;
  }

  /* rotation around X axis */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d x_rotation_mat4d(double theta)
  {
    Mat4d M = Mat4d_identity;
    M.m[0][0] = 1.;
    M.m[1][1] = cos(theta);
    M.m[1][2] = -sin(theta);
    M.m[2][1] = sin(theta);
    M.m[2][2] = cos(theta);
    M.m[3][3] = 1.;
    return M;
  }

  /* rotation around Y axis */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d y_rotation_mat4d(double theta)
  {
    Mat4d M = Mat4d_identity;
    M.m[1][1] = 1.;
    M.m[0][0] = cos(theta);
    M.m[0][2] = sin(theta);
    M.m[2][0] = -sin(theta);
    M.m[2][2] = cos(theta);
    M.m[3][3] = 1.;
    return M;
  }

  /* rotation around X axis */
  ONIKA_HOST_DEVICE_FUNC inline Mat4d z_rotation_mat4d(double theta)
  {
    Mat4d M = Mat4d_identity;
    M.m[2][2] = 1.;
    M.m[0][0] = cos(theta);
    M.m[0][1] = -sin(theta);
    M.m[1][0] = sin(theta);
    M.m[1][1] = cos(theta);
    M.m[3][3] = 1.;
    return M;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline double dot ( const Vec4d& A , const Vec4d& B )
  {
    double r = 0.0;
    for(unsigned int i=0;i<4;i++) { r += A.v[i] * B.v[i]; }
    return r;
  }
  
  ONIKA_HOST_DEVICE_FUNC inline double quadric_eval( const Mat4d& M , const Vec4d& r )
  {
    return dot( r*M , r );
  }

  ONIKA_HOST_DEVICE_FUNC inline double quadric_eval( const Mat4d& M , const Vec3d& r )
  {
    return quadric_eval( M , make_vec4d(r) );
  }

  ONIKA_HOST_DEVICE_FUNC inline bool InvertMatrix(const double m[16], double invOut[16])
  {
    const double inv[16] = {
         m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]
      , -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10]
      ,  m[1] * m[ 6] * m[15] - m[1] * m[ 7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[ 7] - m[13] * m[3] * m[ 6]
      , -m[1] * m[ 6] * m[11] + m[1] * m[ 7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[ 9] * m[2] * m[ 7] + m[ 9] * m[3] * m[ 6]
      , -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10]
      ,  m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10]
      , -m[0] * m[ 6] * m[15] + m[0] * m[ 7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[ 7] + m[12] * m[3] * m[ 6]
      ,  m[0] * m[ 6] * m[11] - m[0] * m[ 7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[ 8] * m[2] * m[ 7] - m[ 8] * m[3] * m[ 6]
      ,  m[4] * m[ 9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[ 9]
      , -m[0] * m[ 9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[ 9]
      ,  m[0] * m[ 5] * m[15] - m[0] * m[ 7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[ 7] - m[12] * m[3] * m[ 5]
      , -m[0] * m[ 5] * m[11] + m[0] * m[ 7] * m[ 9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[ 9] - m[ 8] * m[1] * m[ 7] + m[ 8] * m[3] * m[ 5]
      , -m[4] * m[ 9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[ 9]
      ,  m[0] * m[ 9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[ 9]
      , -m[0] * m[ 5] * m[14] + m[0] * m[ 6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[ 6] + m[12] * m[2] * m[ 5]
      ,  m[0] * m[ 5] * m[10] - m[0] * m[ 6] * m[ 9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[ 9] + m[ 8] * m[1] * m[ 6] - m[ 8] * m[2] * m[ 5] };
    const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det == 0) return false;
    for (unsigned int i = 0; i < 16; i++) invOut[i] = inv[i] / det;
    return true;
  }

  ONIKA_HOST_DEVICE_FUNC inline Mat4d inverse(const Mat4d& M)
  {
    double tmp[16];
    double tmpinv[16];
    for (unsigned int i = 0; i < 16; i++) tmp[i] = M.m[i%4][i/4];
    bool ok = InvertMatrix(tmp,tmpinv);
    Mat4d R = Mat4d_zero;
    if( ok ) for (unsigned int i = 0; i < 16; i++) R.m[i%4][i/4] = tmpinv[i];
    return R;
  }

}

namespace YAML
{

  template<> struct convert< exanb::Mat4d >
  {
    static inline Node encode(const exanb::Mat4d& v)
    {
      Node node;
      for(int j=0;j<4;j++) for(int i=0;i<4;i++) node.push_back( v.m[i][j] );
      return node;
    }
    static inline bool decode(const Node& node, exanb::Mat4d & M)
    {
      //std::cout<<"Mat4d Yaml='";
      //exanb::dump_node_to_stream( std::cout , node );
      //std::cout<<"'"<<std::endl;
      
      if( node.IsSequence() )
      {
        auto lmat = node.as< std::vector<exanb::Mat4d> >();
        M = exanb::Mat4d_identity;
        const unsigned int N = lmat.size();
        for( unsigned int i=0;i<N;i++ )
        {
          M = M * lmat[N-1-i];
          // std::cout << "+TM " << tm <<" -> M = "<<M << std::endl;
        }
      }
      else
      {
        if( node.IsScalar() )
        {
          std::string s = node.as<std::string>();
          if( s == "sphere" ) M = exanb::sphere_quadric();
          else if( s == "conex" ) M = exanb::x_cone_quadric();
          else if( s == "coney" ) M = exanb::y_cone_quadric();
          else if( s == "conez" ) M = exanb::z_cone_quadric();
          else if( s == "cylx" ) M = exanb::x_cylinder_quadric();
          else if( s == "cyly" ) M = exanb::y_cylinder_quadric();
          else if( s == "cylz" ) M = exanb::z_cylinder_quadric();
          else if( s == "planex" ) M = exanb::plane_quadric( Plane3d{ Vec3d{1,0,0} , 0 } );
          else if( s == "planey" ) M = exanb::plane_quadric( Plane3d{ Vec3d{0,1,0} , 0 } );
          else if( s == "planez" ) M = exanb::plane_quadric( Plane3d{ Vec3d{0,0,1} , 0 } );
          else return false;
        }
        else if( node.IsMap() )
        {
          if( node["xrot"] ) M = exanb::x_rotation_mat4d( node["xrot"].as<Quantity>().convert() );
          else if( node["yrot"] ) M = exanb::y_rotation_mat4d( node["yrot"].as<Quantity>().convert() );
          else if( node["zrot"] ) M = exanb::z_rotation_mat4d( node["zrot"].as<Quantity>().convert() );
          else if( node["scale"] ) M = exanb::scaling_mat4d( node["scale"].as<Vec3d>() );
          else if( node["translate"] ) M = exanb::translation_mat4d( node["translate"].as<Vec3d>() );
          else if( node["plane"] ) M = exanb::plane_quadric( node["plane"].as<Plane3d>() );
          else { return false; }
        }
        else
        {
          std::cerr<<"Mat4d conversion error: Node is neither map or scalar. node content :"<<std::endl;
          exanb::dump_node_to_stream( std::cerr , node );
          std::cerr<<std::endl;
          return false;
        }
      }
      //std::cout << std::setprecision(3) << "MAT="<< M << std::endl;
      return true;
    }
  };  

}


// =========== Unit tests ====================
#include <onika/test/unit_test.h>
#include <random>

ONIKA_UNIT_TEST(matrix_4d)
{
  using namespace exanb;

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<double> ud( -1 , 1 );

  const auto M1 = YAML::Load( R"EOF(
- translate: [ 1 , 2 , -3 ]
- scale: [ 2.0 , 0.25 , 0.5 ]
)EOF").as<exanb::Mat4d>();

  const auto M2 = YAML::Load( R"EOF(
- translate: [ 1 , 2 , -3 ]
- scale: [ 2.0 , 0.25 , 0.5 ]
- translate: [ -3 , 2 , 1 ]
)EOF").as<exanb::Mat4d>();

  const Vec3d T = { 1 , 2 , -3 };
  const Vec3d S = { 2.0 , 0.25 , 0.5 };
  const Vec3d T2 = { -3 , 2 , 1 };
  std::vector<Vec3d> points = { {0,0,0} , {1,1,1} , {-1,-1,-1} };

  const auto M1_inv = inverse(M1);
  const auto M2_inv = inverse(M2);

//  std::cout <<"M1="<<M1 <<std::endl;
//  std::cout <<"M2="<<M2 <<std::endl;
//  std::cout <<"transpose(M2)="<<transpose(M2) <<std::endl;

  for(unsigned int i=0;i<1000;i++)
  {
    Vec3d p = { ud(gen) , ud(gen) , ud(gen) };
    if(i<points.size()) { p = points[i]; }
    const auto tp1 = make_vec3d( M1 * make_vec4d(p) );
    const auto tp2 = make_vec3d( M2 * make_vec4d(p));
    const double err1 = norm( ( (p+T)*S ) - tp1 );
    const double err2 = norm( ( ((p+T)*S)+T2 ) - tp2 );
    if( err1 >= 1.e-10 || err2 >= 1.e-10 )
    {
      std::cerr <<"P=("<<p << ") , (P+T)*S=("<<(p+T)*S << ") , TP1=("<<tp1 << ") , E1="<<err1<<" , ((P+T)*S)+T2=("<<((p+T)*S)+T2 <<") , TP2=("<<tp2<<") , E2="<<err2<<std::endl;
    }
    ONIKA_TEST_ASSERT( err1 < 1.e-10 && err2 < 1.e-10 );
    const auto itp1 = make_vec3d( M1_inv * make_vec4d(tp1) );
    const auto itp2 = make_vec3d( M2_inv * make_vec4d(tp2) );
    const double ierr1 = norm( p - itp1 );
    const double ierr2 = norm( p - itp2 );
    if( ierr1 >= 1.e-10 || ierr2 >= 1.e-10 )
    {
      std::cerr << "M1_inv*TP1="<<itp1<<" , M2_inv*TP2="<<itp2<<std::endl;
    }
    ONIKA_TEST_ASSERT( ierr1 < 1.e-10 && ierr2 < 1.e-10 );
  }
}

