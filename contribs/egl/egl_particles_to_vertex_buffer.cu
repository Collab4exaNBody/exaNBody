
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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>
#include <onika/math/basic_types_def.h>
#include <onika/math/quaternion.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/field_combiners.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/particle_type_properties.h>
#include <exanb/core/grid_additional_fields.h>

#include <EGLRender/egl_render_manager.h>

namespace exanb
{
  using namespace EGLRender;

  using Vec3d = exanb::Vec3d;
  using Quat = exanb::Quaternion;
  using Mat3d = exanb::Mat3d;

  template<class T> struct GLTypeId
  {
    using comp_type = T;
    static inline constexpr GLenum type_enum = GL_NONE;
    static constexpr int ncomp = 0;
    static inline std::string str() { return "void"; }
  };

template<class A, class V> struct GLAttributeWriter { ONIKA_HOST_DEVICE_FUNC static inline void write(A* a, const V& v) { *a = v; } };
template<> struct GLAttributeWriter<GLfloat,Vec3d>  { ONIKA_HOST_DEVICE_FUNC static inline void write(GLfloat* a, const Vec3d& v) { a[0]=v.x; a[1]=v.y; a[2]=v.z; } };
template<> struct GLAttributeWriter<GLfloat,Quat>   { ONIKA_HOST_DEVICE_FUNC static inline void write(GLfloat* a, const Quat& v) { a[0]=v.x; a[1]=v.y; a[2]=v.z; a[3]=v.w; } };
template<> struct GLAttributeWriter<GLfloat,Mat3d>  { ONIKA_HOST_DEVICE_FUNC static inline void write(GLfloat* a, const Mat3d& v)
  { a[0]=v.m11; a[1]=v.m12; a[2]=v.m13;
    a[3]=v.m21; a[4]=v.m22; a[5]=v.m23;
    a[6]=v.m31; a[7]=v.m32; a[8]=v.m33; } };
template<size_t N> struct GLAttributeWriter<GLfloat , onika::oarray_t<double,N> > { ONIKA_HOST_DEVICE_FUNC static inline void write(GLfloat * a, const onika::oarray_t<double,N> & v) { for(size_t i=0;i<N;i++) a[i]=v[i]; } };
template<size_t N> struct GLAttributeWriter<GLfloat ,    std::array  <double,N> > { ONIKA_HOST_DEVICE_FUNC static inline void write(GLfloat * a, const    std::array  <double,N> & v) { for(size_t i=0;i<N;i++) a[i]=v[i]; } };

#define GLTypeInfoMacro(rtype,gltype,glenum,nc,tname) \
  template<> struct GLTypeId<rtype> { \
    using comp_type=gltype; \
    static inline constexpr GLenum type_enum=glenum; \
    static inline constexpr int ncomp=nc; \
    static inline std::string str() { return tname; } };

  GLTypeInfoMacro( uint8_t , GLubyte  , GL_UNSIGNED_BYTE ,1,"uint")
  GLTypeInfoMacro(  int8_t , GLbyte   , GL_BYTE          ,1,"int")
  GLTypeInfoMacro(uint16_t , GLushort , GL_UNSIGNED_SHORT,1,"uint")
  GLTypeInfoMacro( int16_t , GLshort  , GL_SHORT         ,1,"int")
  GLTypeInfoMacro(uint32_t , GLuint   , GL_UNSIGNED_INT  ,1,"uint")
  GLTypeInfoMacro( int32_t , GLint    , GL_INT           ,1,"int")
  GLTypeInfoMacro(uint64_t , GLuint   , GL_UNSIGNED_INT  ,1,"uint")
  GLTypeInfoMacro( int64_t , GLint    , GL_INT           ,1,"int")
  GLTypeInfoMacro( float   , GLfloat  , GL_FLOAT         ,1,"float")
  GLTypeInfoMacro( double  , GLfloat  , GL_FLOAT         ,1,"float")
  GLTypeInfoMacro( Vec3d   , GLfloat  , GL_FLOAT         ,3,"vec3")
  GLTypeInfoMacro( Quat    , GLfloat  , GL_FLOAT         ,4,"vec4")
  GLTypeInfoMacro( Mat3d   , GLfloat  , GL_FLOAT         ,9,"mat3")

# undef GLTypeInfoMacro

  template<class T, size_t N> struct GLTypeId< std::array<T,N> >
  {
    using comp_type = typename GLTypeId<T>::comp_type;
    static inline constexpr GLenum type_enum = GLTypeId<T>::type_enum;
    static constexpr int ncomp = N;
    static inline std::string str() { return GLTypeId<T>::str() +"["+std::to_string(N)+"]"; }
  };

  template<class T, size_t N> struct GLTypeId< onika::oarray_t<T,N> >
  {
    using comp_type = typename GLTypeId<T>::comp_type;
    static inline constexpr GLenum type_enum = GLTypeId<T>::type_enum;
    static constexpr int ncomp = N;
    static inline std::string str() { return GLTypeId<T>::str() +"["+std::to_string(N)+"]"; }
  };

  template<class CellsT, class FieldT>
  struct GLVertexAttribCopyFromParticles
  {
    using field_type = typename FieldT::value_type;
    using GLTypeIdT = GLTypeId<field_type>;
    using gl_comp_type = typename GLTypeIdT::comp_type;
    using Writer = GLAttributeWriter<gl_comp_type,field_type>;

    // particle field access
    CellsT m_cells;
    FieldT m_field;
    const size_t * m_cell_particle_offset = nullptr;

    // GL buffer access
    gl_comp_type * m_attrib_ptr = nullptr;

    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t cell_i, unsigned int p_i, const field_type& p_attr ) const
    {
      Writer::write( m_attrib_ptr + ( ( m_cell_particle_offset[cell_i] + p_i ) * GLTypeIdT::ncomp ) , m_cells[cell_i][m_field][p_i] );
    }

  };

  template<class CellsT, class FieldT> struct ComputeCellParticlesTraits< exanb::GLVertexAttribCopyFromParticles<CellsT,FieldT> >
  {
    static inline constexpr bool CudaCompatible = true;
  };

  template<class GridT>
  class EGLParticlesToVertexBuffer : public OperatorNode
  {
    using StringIntMap = std::map< std::string , int >;

    ADD_SLOT( GridT            , grid               , INPUT_OUTPUT , DocString{"Local sub-domain particles grid"} );
    ADD_SLOT( ParticleTypeProperties , particle_type_properties , INPUT , OPTIONAL );
    ADD_SLOT( StringIntMap     , vertex_attribs     , INPUT , StringIntMap() , DocString{"Mapping of fields to vertex attribute indices"} );
    ADD_SLOT( std::string      , vertex_buffer      , INPUT_OUTPUT , "particles" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:

    template<class FieldT>
    inline void process_one_field( GLVertexBuffers & glvbos, const FieldT& f )
    {
      using field_type = typename FieldT::value_type;
      using attrib_type = typename GLTypeId<field_type>::comp_type;
      auto it = vertex_attribs->find( f.name() );
      if( it != vertex_attribs->end() )
      {
        const int ai = it->second;
        ldbg << "write field "<<f.name()<<" to attrib #"<<ai<<std::endl;
        const auto cells = grid->cells_accessor();
        using CellsT = std::remove_cv_t< std::remove_reference_t< decltype(cells) > >;
        bool runs_on_gpu = ( global_cuda_ctx()!=nullptr && global_cuda_ctx()->has_devices() );

        attrib_type * attrib_ptr = nullptr;
        if( runs_on_gpu ) attrib_ptr = (attrib_type*) glvbos.gpu_map_write_only(ai);
        else attrib_ptr = (attrib_type*) glvbos.host_map_write_only(ai);

        GLVertexAttribCopyFromParticles<CellsT,FieldT> cp_func = { cells, f, grid->cell_particle_offset_data(), attrib_ptr };

        auto cp_fields = onika::make_flat_tuple(f);
        compute_cell_particles( *grid , false, cp_func, cp_fields, parallel_execution_context("CopyGLVertAttr") );

        if( runs_on_gpu ) glvbos.gpu_unmap(ai);
        else glvbos.host_unmap(ai);
      }
    }
    template<class FieldT>
    inline void process_one_field( GLVertexBuffers & glvbos, const std::span<FieldT>& fvec )
    {
      for(const auto& f : fvec) process_one_field( glvbos, f );
    }


    template<class FieldT>
    inline void build_formats(std::map<GLint , std::pair<GLenum,GLint> > & attrib_formats, const FieldT& f )
    {
      using field_type = typename FieldT::value_type;
      auto it = vertex_attribs->find( f.name() );
      if( it != vertex_attribs->end() )
      {
        const int ai = it->second;
        attrib_formats[ai].first = GLTypeId<field_type>::type_enum;
        attrib_formats[ai].second = GLTypeId<field_type>::ncomp;
      }
    }
    template<class FieldT>
    inline void build_formats( std::map<GLint , std::pair<GLenum,GLint> > & attrib_formats, const std::span<FieldT>& fvec )
    {
      for(const auto& f : fvec) build_formats( attrib_formats , f );
    }

    template<class... GridFields>
    inline void execute_on_fields( const GridFields& ... grid_fields )
    {
      const size_t n_points = grid->number_of_particles();
      ldbg << "total particles = "<<n_points <<std::endl;

      int buf_id = egl_render_manager->vertex_buffers_id( *vertex_buffer );
      if( buf_id == -1 )
      {
        ldbg << "EGL : create vertex buffer " << *vertex_buffer <<std::endl;
        std::map<int , std::pair<GLenum,GLint> > attrib_formats;
        ( ... , ( build_formats(attrib_formats,grid_fields) ) ) ;
        if(attrib_formats.empty()) return;
        const int nb_attribs = 1 + attrib_formats.crbegin()->first;
        ldbg << "nb_attribs = " << nb_attribs << std::endl;
        std::vector<GLint> num_attribs( nb_attribs*2 , GL_NONE );
        for(const auto& p : attrib_formats)
        {
          ldbg << "insert attrib @"<<p.first<<" enum="<<gl_enum_to_string(p.second.first)<<", ncomp="<< p.second.second<<std::endl;
          num_attribs[ p.first *2 ] = p.second.first;
          num_attribs[ p.first *2 +1 ] = p.second.second;
        }
        for(size_t i=0;i<num_attribs.size()/2;i++)
        {
          if( num_attribs[i*2]==GL_NONE || num_attribs[i*2+1]<1 ||num_attribs[i*2+1]>9 )
          {
            fatal_error() << "Invalid Attribute mapping at index #"<<i <<" , enum="<< gl_enum_to_string(num_attribs[i*2])<<" , ncomp="<< num_attribs[i*2+1] << std::endl;
          }
        }
        buf_id = egl_render_manager->create_vertex_buffers( *vertex_buffer , n_points , num_attribs );
      }

      GLVertexBuffers & glvbos = egl_render_manager->vertex_buffers(buf_id);
      ldbg << "EGL : update vertex buffer " << *vertex_buffer << " , nv="<< n_points << " , id="<<buf_id<<std::endl;
      glvbos.set_number_of_vertices( n_points );

      ( ... , ( process_one_field(glvbos,grid_fields) ) ) ;

      parallel_execution_queue().wait();
    }

    template<class... fid>
    inline void execute_on_field_set( FieldSet<fid...> )
    {
      using has_field_type_t = typename GridT:: template HasField < field::_type >;
      static constexpr bool has_field_type = has_field_type_t::value;

      PositionVec3Combiner position = {};
      VelocityVec3Combiner velocity = {};
      ForceVec3Combiner    force    = {};

      // optional fields
      ParticleTypeProperties * optional_type_properties = nullptr;
      if ( has_field_type && particle_type_properties.has_value() ) optional_type_properties = particle_type_properties.get_pointer();
      GridAdditionalFields add_fields( grid , optional_type_properties );
      auto [ type_real_fields, type_vec3_fields, type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields ] = add_fields.view();

      // remove rx,ry,rz, fx,fy,fz and vx,vy,vz
      execute_on_fields( position, velocity, force, type_real_fields, type_vec3_fields, type_mat3_fields, opt_real_fields, opt_vec3_fields, opt_mat3_fields, onika::soatl::FieldId<fid>{} ... );
    }

    inline void execute() override final
    {
      using GridFieldSet = RemoveFields< typename GridT::field_set_t , FieldSet< field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz> >;
      execute_on_field_set( GridFieldSet{} );
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_particles_to_vertex_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_particles_to_vertex_buffer", make_grid_variant_operator< EGLParticlesToVertexBuffer > );
  }

}

