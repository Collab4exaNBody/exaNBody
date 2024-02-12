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
#include <algorithm>
#include <onika/stream_utils.h>

namespace onika
{

  struct RGBColord
  {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
  };

  struct RGBAColord
  {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    double a = 1.0;
  };

  struct RGBColor8
  {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
  };

  struct RGBAColor8
  {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
  };

  extern RGBColord cielab_colormap(double x);
  extern RGBAColord cielab_colormap(double x, double a);

  inline RGBColord cielab_discreet_colormap(int i, int N) { return cielab_colormap(i*1.0/N); }
  inline RGBAColord cielab_discreet_colormap(int i, int N, double a){ return cielab_colormap(i*1.0/N,a); }

  extern RGBColor8 to_rgb8(const RGBColord& c);
  extern RGBAColor8 to_rgba8(const RGBAColord& c);

  extern RGBColord alpha_multiply(const RGBAColord& c);

  inline RGBColord operator * (const RGBColord& c, double x)
  {
    return { c.r*x , c.g*x , c.b*x };
  }

  template<class StreamT>
  inline StreamT& color_to_stream_www(StreamT& out, const RGBColor8 & c)
  {
    static const char* HEX = "0123456789ABCDEF";
    auto [r,g,b] = c;
    return out << '#' << HEX[r/16] << HEX[r%16] << HEX[g/16] << HEX[g%16] << HEX[b/16] << HEX[b%16] ;
  }
  template<class StreamT> inline StreamT& color_to_stream_www(StreamT& out, const RGBColord & c) { return color_to_stream_www(out,to_rgb8(c)); }

  template<class StreamT>
  inline StreamT& color_to_stream_latex(StreamT& out, const RGBColord & c)
  {
    return out << "{rgb}{"<<c.r<<','<<c.g<<','<<c.b<<"}" ;
  }

  template<class StreamT>
  inline StreamT& color_to_stream_www(StreamT& out, const RGBAColor8 & c)
  {
    static const char* HEX = "0123456789ABCDEF";
    auto [r,g,b,a] = c;
    return out << '#' << HEX[r/16] << HEX[r%16] << HEX[g/16] << HEX[g%16] << HEX[b/16] << HEX[b%16] << HEX[a/16] << HEX[a%16];
  }
  template<class StreamT> inline StreamT& color_to_stream_www(StreamT& out, const RGBAColord & c) { return color_to_stream_www(out,to_rgba8(c)); }



  // color formatting suitable for html and other formats, such as X11
  template<class ColorT>
  struct WWWPrintableColor
  {
    ColorT m_color;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return color_to_stream_www(out,m_color); }
  };

  template<class ColorT>
  struct PrintableFormattedObject< WWWPrintableColor<ColorT> >
  {
    WWWPrintableColor<ColorT> m_www_color;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return m_www_color.to_stream(out); }
  };
  template<class ColorT>
  inline PrintableFormattedObject< WWWPrintableColor<ColorT> > format_color_www(const ColorT& c) { return { {c} } ; }


  // color formatting suitable for LateX
  template<class ColorT>
  struct LatexPrintableColor
  {
    ColorT m_color;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return color_to_stream_latex(out,m_color); }
  };

  template<class ColorT>
  struct PrintableFormattedObject< LatexPrintableColor<ColorT> >
  {
    LatexPrintableColor<ColorT> m_color;
    template<class StreamT> inline StreamT& to_stream(StreamT& out) const { return m_color.to_stream(out); }
  };

  template<class ColorT>
  inline PrintableFormattedObject< LatexPrintableColor<ColorT> > format_color_tex(const ColorT& c) { return { {c} } ; }

}

