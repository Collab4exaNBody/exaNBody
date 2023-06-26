//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <onika/memory/allocator.h>

#include <onika/soatl/field_pointer_tuple.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>

#include <memory>

namespace exanb
{

  struct InPlaceAddFunctor
  {
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( T& a , const T& b ) const { a += b; }
  };

  struct InPlaceMulFunctor
  {
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( T& a , const T& b ) const { a *= b; }
  };

  struct SetFirstArgFunctor
  {
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( T& a, const T& b ) const { a = b; }
  };

  template<class OP>
  struct GenericVec3Functor
  {
    const Vec3d v;
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double & x, double & y, double & z ) const
    {
      Vec3d r = {x,y,z};
      op( r , v );
      x = r.x;
      y = r.y;
      z = r.z;
    }
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericVec3Functor<OP> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<class OP>
  struct GenericVec3RegionFunctor
  {
    const ParticleRegionCSGShallowCopy region;
    const Vec3d v;
    const OP op;
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double rx, double ry, double rz, const uint64_t id, double & x, double & y, double & z ) const
    {
      Vec3d r = {rx,ry,rz};
      if( region.contains( r , id ) )
      {
        Vec3d VEC = {x,y,z};
        op( VEC , v );
        x = VEC.x;
        y = VEC.y;
        z = VEC.z;
      }
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double rx, double ry, double rz, double & x, double & y, double & z ) const
    {
      this->operator () (rx,ry,rz,0,x,y,z);
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( const uint64_t id, double & x, double & y, double & z ) const
    {
      this->operator () (x,y,z,id,x,y,z);
    }    
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( double & x, double & y, double & z ) const
    {
      this->operator () (x,y,z,0,x,y,z);
    }    
  };

  template<class OP> struct ComputeCellParticlesTraits< GenericVec3RegionFunctor<OP> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<
    class GridT,
    class Field_X, class Field_Y, class Field_Z,
    class OpT ,
    class = AssertGridHasFields< GridT, Field_X, Field_Y, Field_Z >
    >
  class GenericVec3Operator : public OperatorNode
  {  
    ADD_SLOT( Vec3d , value , INPUT , REQUIRED );
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT);

    ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
    ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL );

    static constexpr OpT Func = OpT{};

    static constexpr FieldSet< Field_X, Field_Y, Field_Z > compute_field_set{};
    using has_field_id_t     = typename GridT:: template HasField <field::_id>;
    static constexpr bool has_field_id = has_field_id_t::value;
    static constexpr bool has_separate_r_fields = ! ( std::is_same_v<Field_X,field::_rx> && std::is_same_v<Field_Y,field::_ry> && std::is_same_v<Field_Z,field::_rz> );
    
    using compute_field_set_region_t = 
      std::conditional_t< has_field_id ,      
        std::conditional_t< has_separate_r_fields ,
          FieldSet< field::_rx , field::_ry , field::_rz , field::_id, Field_X, Field_Y, Field_Z >
          ,
          FieldSet< field::_id, Field_X, Field_Y, Field_Z > >
        ,
        std::conditional_t< has_separate_r_fields ,
          FieldSet< field::_rx , field::_ry , field::_rz , Field_X, Field_Y, Field_Z >
          ,
          FieldSet< Field_X, Field_Y, Field_Z > > 
        >;
    static constexpr compute_field_set_region_t compute_field_set_region{};

  public:
    inline void execute () override final
    {
      ldbg<<"GenericVec3Operator: value="<<(*value)<<std::endl;

      if( grid->number_of_cells() == 0 ) return;

      if( region.has_value() )
      {
        if( !particle_regions.has_value() )
        {
          fatal_error() << "GenericVec3Operator: region is defined, but particle_regions has no value" << std::endl;
        }        
        if( region->m_nb_operands==0 )
        {
          ldbg << "rebuild CSG from expr "<< region->m_user_expr << std::endl;
          region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
        }
        ParticleRegionCSGShallowCopy prcsg = *region;
        ldbg << "Particle Region CSG\n\tm_expr = "<<prcsg.m_expr<<std::endl
             << "\tm_nb_regions = " << static_cast<int>(prcsg.m_nb_regions)<<std::endl
             << "\tm_nb_operands = " << static_cast<int>(prcsg.m_nb_operands)<<std::endl
             << "\tm_nb_operands_log2 = " << static_cast<int>(prcsg.m_nb_operands_log2)<<std::endl
             << "\tm_operand_places =";
        for(unsigned int i=0;i<prcsg.m_nb_operands;i++) ldbg<<" "<<static_cast<int>(prcsg.m_operand_places[i]);
        ldbg << std::endl << "\tm_regions = " << std::endl;
        for(unsigned int i=0;i<prcsg.m_nb_regions;i++)
        {
          const auto & R = prcsg.m_regions[i];
          ldbg << "\t\t"<< R.m_name << std::endl;
          ldbg << "\t\t\tQuadric = "<< R.m_quadric << std::endl;
          ldbg << "\t\t\tBounds  = "<< R.m_bounds << std::endl;
          ldbg << "\t\t\tId range = [ "<< R.m_id_start << " ; "<<R.m_id_end<<" [" << std::endl;
        }

        compute_cell_particles( *grid , false , GenericVec3RegionFunctor<OpT>{prcsg,*value,Func} , compute_field_set_region , gpu_execution_context() , gpu_time_account_func() );            
      }
      else
      {
        compute_cell_particles( *grid , false , GenericVec3Functor<OpT>{*value,Func} , compute_field_set , gpu_execution_context() , gpu_time_account_func() );            
      }
      
    }

    inline void yaml_initialize(const YAML::Node& node) override final
    {
      YAML::Node tmp;
      if( node.IsSequence() )
      {
        tmp["value"] = node;
      }
      else if( node.IsScalar() )
      {
        double x = node.as<Quantity>().convert();
        tmp["value"] = std::vector<double> { x , x , x };
      }
      else { tmp = node; }
      this->OperatorNode::yaml_initialize( tmp );
    }

  };

  template<class GridT> using ShiftPositionOperator = GenericVec3Operator< GridT, field::_rx,field::_ry,field::_rz , InPlaceAddFunctor >;
  template<class GridT> using ShiftVelocityOperator = GenericVec3Operator< GridT, field::_vx,field::_vy,field::_vz , InPlaceAddFunctor >;
  template<class GridT> using ScalePositionOperator = GenericVec3Operator< GridT, field::_rx,field::_ry,field::_rz , InPlaceMulFunctor >;
  template<class GridT> using SetVelocityOperator   = GenericVec3Operator< GridT, field::_vx,field::_vy,field::_vz , SetFirstArgFunctor >;
  template<class GridT> using SetForceOperator      = GenericVec3Operator< GridT, field::_fx,field::_fy,field::_fz , SetFirstArgFunctor >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "shift_r", make_grid_variant_operator< ShiftPositionOperator > );
   OperatorNodeFactory::instance()->register_factory( "shift_v", make_grid_variant_operator< ShiftVelocityOperator > );
   OperatorNodeFactory::instance()->register_factory( "scale_r", make_grid_variant_operator< ScalePositionOperator > );
   OperatorNodeFactory::instance()->register_factory( "set_velocity", make_grid_variant_operator< SetVelocityOperator > );
   OperatorNodeFactory::instance()->register_factory( "set_force", make_grid_variant_operator< SetForceOperator > );
  }

}

