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
#include <onika/color_scale.h>
#include <cstdint>

namespace onika
{

  // x must be in [0.;1.]
  RGBColord cielab_colormap(double x)
  {
    static const double endpoints[][3] =
      { {0.230, 0.299, 0.754}     // #3A4CC0
      //, {0.8654, 0.8654, 0.8654}  // #DCDCDC
      , {0.75, 0.75, 0.75}  // #C0C0C0
      , {0.706, 0.016, 0.150} };  // #B40426
    double t = std::clamp(x,0.0,1.0);
    int s = 0, e = 1;
    if( t > 0.5 ) { s=1; e=2; t-=0.5; }
    t *= 2.0;
    double R = endpoints[s][0]*(1.0-t) + endpoints[e][0] *t;
    double G = endpoints[s][1]*(1.0-t) + endpoints[e][1] *t;
    double B = endpoints[s][2]*(1.0-t) + endpoints[e][2] *t;
    return {R,G,B};
  }

  RGBColor8 to_rgb8(const RGBColord& c)
  {
    auto [R,G,B] = c;
    uint8_t r = std::clamp( int(R*255) , 0 , 255 );
    uint8_t g = std::clamp( int(G*255) , 0 , 255 );
    uint8_t b = std::clamp( int(B*255) , 0 , 255 );
    return {r,g,b};
  }

  RGBAColord cielab_colormap(double x, double a)
  {
    auto [r,g,b] = cielab_colormap(x);
    return {r,g,b,a};
  }
  
  RGBAColor8 to_rgba8(const RGBAColord& c)
  {
    auto [R,G,B,A] = c;
    uint8_t r = std::clamp( int(R*255) , 0 , 255 );
    uint8_t g = std::clamp( int(G*255) , 0 , 255 );
    uint8_t b = std::clamp( int(B*255) , 0 , 255 );
    uint8_t a = std::clamp( int(A*255) , 0 , 255 );
    return {r,g,b,a};
  }

  RGBColord alpha_multiply(const RGBAColord& c)
  {
    auto [r,g,b,a] = c;
    return {r*a,g*a,b*a};
  }

}

