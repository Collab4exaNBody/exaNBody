#pragma once 

#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/fields.h>

//#include "exanb/general_matrix_algebra/mat3d_bridge.h"
//#include "exanb/general_matrix_algebra/matrix.h"

//#include <memory>
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <string>
#include <iomanip>
#include <experimental/filesystem>
#include <math.h>

namespace exanb {

  static inline int sign(const double& test) {return (test >= 0) - (test < 0);}

  //fonction d'atténuation de Wendland normalisée à 10
  static inline double Wendland_C2(const double& x) { return (1+2*x)*pow(1-x/2., 4.);  }

  static inline double Wendland_weight(const double& dist, const double& DistWeight) {
  
    double sigma = DistWeight/3.;

    if(dist < 4*sigma) {    
      double alpha = 10*21/16./M_PI;
      double beta2 = 4/15.;

      return alpha*beta2/pow(sigma, 3.)*Wendland_C2(dist/sigma*sqrt(beta2));
    }

    else 
      return 0.;
  }
 
  //fonction d'atténuation "maison" non normalisée
  static inline double calc_weight(const double& DistCur, const double& DistWeight) {
	  
    if(DistCur/(DistWeight) <= 0.5) {
	  return 1.0 - 6.0 * pow((DistCur/(DistWeight)),2.0) + 6.0 * pow((DistCur/(DistWeight)),3.0);
	}
	else if(((DistCur/(DistWeight)) > 0.5) && ((DistCur/(DistWeight)) < 1.0)) {
	  return 2.0 - 6.0 * (DistCur/(DistWeight)) + 6.0 * pow((DistCur/(DistWeight)),2.0) - 2.0 * pow((DistCur/(DistWeight)),3.0);
	}

    return 0;
  }

  static inline void deal_with_periodicity_ref_pos(Vec3d & RelativPosRef, const Mat3d & box_normalizer, Vec3d & ref_periodicity_signature) {
 
    Vec3d deltaPosRefCris;
    deltaPosRefCris = inverse(box_normalizer) * RelativPosRef;
          
    deltaPosRefCris.x += ref_periodicity_signature.x;
    deltaPosRefCris.y += ref_periodicity_signature.y;
    deltaPosRefCris.z += ref_periodicity_signature.z;
          
    if(deltaPosRefCris.x > 0.5) {
      deltaPosRefCris.x -= 1.0;
      ref_periodicity_signature.x -= 1; 
    } else if(deltaPosRefCris.x < -0.5) {
      deltaPosRefCris.x += 1.0;
      ref_periodicity_signature.x += 1; 
    }

    if(deltaPosRefCris.y > 0.5) {
      deltaPosRefCris.y -= 1.0;
      ref_periodicity_signature.y -= 1; 
    } else if(deltaPosRefCris.y < -0.5) {
      deltaPosRefCris.y += 1.0;
      ref_periodicity_signature.y += 1; 
    }

    if(deltaPosRefCris.z > 0.5) {
      deltaPosRefCris.z -= 1.0;
      ref_periodicity_signature.z -= 1; 
    } else if(deltaPosRefCris.z < -0.5) {
      deltaPosRefCris.z += 1.0;
      ref_periodicity_signature.z += 1; 
    }

    RelativPosRef = box_normalizer * deltaPosRefCris;
  }


  static inline void deal_with_periodicity_cur_pos(Vec3d & RelativPosCur, const Mat3d & box_normalizer, const Vec3d & ref_periodicity_signature) {
 
    Vec3d deltaPosCourCris;
    deltaPosCourCris = inverse(box_normalizer) * RelativPosCur;
          
    deltaPosCourCris.x += ref_periodicity_signature.x;
    deltaPosCourCris.y += ref_periodicity_signature.y;
    deltaPosCourCris.z += ref_periodicity_signature.z;
          
    if(deltaPosCourCris.x > 0.5) {
      deltaPosCourCris.x -= 1.0;
    } else if(deltaPosCourCris.x < -0.5) {
      deltaPosCourCris.x += 1.0;
    }

    if(deltaPosCourCris.y > 0.5) {
      deltaPosCourCris.y -= 1.0;
    } else if(deltaPosCourCris.y < -0.5) {
      deltaPosCourCris.y += 1.0;
    }

    if(deltaPosCourCris.z > 0.5) {
      deltaPosCourCris.z -= 1.0;
    } else if(deltaPosCourCris.z < -0.5) {
      deltaPosCourCris.z += 1.0;
    }

    RelativPosCur = box_normalizer * deltaPosCourCris;
  }


  template< class GridT >
    struct MechanicalMeasureComputeOp
    {
      using CellsT = decltype( GridT{}.cells() );
      const CellsT m_cells_t0;
     
      //------- Tables de mesures du premier ordre à remplir

      //tables de mesures statiques
      std::vector< std::vector<Mat3d> > & transformation_gradient_tensor_data;
      std::vector< std::vector<Mat3d> > & green_lagrange_tensor_data;
      std::vector< std::vector<Mat3d> > & rotation_tensor_data;
      std::vector< std::vector<Mat3d> > & pure_deformation_tensor_data;
      std::vector< std::vector<Vec3d> > & microrotation_vector_data;
      std::vector< std::vector<Vec3d> > & slip_vector_data;

      std::vector< std::vector<Vec3d> > & l_vector_data;
      std::vector< std::vector<Vec3d> > & m_vector_data;
      std::vector< std::vector<Vec3d> > & n_vector_data;

      //tables de mesures cinématiques
      std::vector< std::vector<Mat3d> > & speed_gradient_tensor_data;
      std::vector< std::vector<Vec3d> > & vortex_vector_data;

      bool compute_static_measure;
      bool compute_cinematic_measure;

      const Mat3d m_xform_t0;
      const Mat3d m_xform;
      const Mat3d m_lattice;
      double m_rrDef;

      double (*weight_function)(const double&, const double&) = calc_weight;
      

      template<class ComputeBufferT, class CellParticlesT, class GridParticleLocksT, class ParticleLockT>
      inline void operator ()
        (
        // number of neighbors
        size_t n,

        // buffer holding attached data to particle's neighbors
        const ComputeBufferT& tab,
                
        // data and locks accessors for neighbors
        CellParticlesT* cells,
        GridParticleLocksT locks,
        ParticleLockT & particle_lock
        )
      {
        //----------- objets objectifs temporaires
        
        //static measures
        Mat3d transformation_gradient_tensor; // intitialized to all zero
        Mat3d green_lagrange_tensor; // intitialized to all zero
        Mat3d rotation_tensor;
        Mat3d pure_deformation_tensor;
        Vec3d microrotation_vector;
        Vec3d slip_vector;

        //cinematic measures
        Mat3d speed_gradient_tensor; // intitialized to all zero
        Vec3d vortex_vector; // intitialized to all zero
	
        //slip basis
        Vec3d l_vector;
	    Vec3d m_vector;
	    Vec3d n_vector;
        

        //----------- constructeurs

        //F, L constructors
        Mat3d ref_tens_ref_position_tensor; // intitialized to all zero
        Mat3d inv_ref_tens_ref_position_tensor; // intitialized to all zero        
        Mat3d cur_tens_ref_position_tensor;       
        Mat3d cur_tens_cur_position_tensor; 
        Mat3d inv_cur_tens_cur_position_tensor; // intitialized to all zero        
        Mat3d speed_tens_cur_position_tensor;
        
        Vec3d deltaPosInit;
        Vec3d deltaPosCour;
        Vec3d deltaPosInitCris;
        Vec3d deltaPosCourCris;

        //slip vector constructors
        Vec3d deltaCourInit; //deplacement des voisins
        int slipped_neighbor_nbr=0;

	    //slip_basis_constuctors
	    Mat3d l_constructor_tensor;
	    Mat3d n_constructor_tensor;
        double square_l;


        //---------- Measures Computation
        
        //initial position computation in the right frame 
        Vec3d dtr;

        Mat3d hht = m_xform * m_lattice;
        Mat3d hh0 = m_xform_t0 * m_lattice;

    	Vec3d tmp;
	    tmp.x = m_cells_t0[tab.cell][field::rx][tab.part];
	    tmp.y = m_cells_t0[tab.cell][field::ry][tab.part];
	    tmp.z = m_cells_t0[tab.cell][field::rz][tab.part];

	    Vec3d PosInit = m_xform_t0 * tmp;

	    Vec3d PosInitVois;

	    Mat3d I = make_identity_matrix();

        Vec3d deltaVit;

	    // loop over neighborhood
        for(size_t i=0;i<n;i++)
        {

          //relativ positions computation
          tmp.x = tab.ext.rx0[i];
          tmp.y = tab.ext.ry0[i];
          tmp.z = tab.ext.rz0[i];

	      PosInitVois = m_xform_t0 * tmp;

	      deltaPosInit = PosInitVois - PosInit;
          
          deltaPosCour.x = tab.drx[i];
          deltaPosCour.y = tab.dry[i];
          deltaPosCour.z = tab.drz[i];


          dtr.x = dtr.y = dtr.z = 0.0;           
          deal_with_periodicity_ref_pos(deltaPosInit, hh0, dtr);
          deal_with_periodicity_cur_pos(deltaPosCour, hht, dtr);

          //weight function
          double poidsD;
          double rrInit2 = norm2(deltaPosInit);
          double rrCour2 = norm2(deltaPosCour);

          poidsD = weight_function(sqrt(rrInit2), m_rrDef);


          if(compute_static_measure) {

            //F constructors assembly
            if(rrInit2 <= 1.2*m_rrDef*m_rrDef) {

              ref_tens_ref_position_tensor += tensor(deltaPosInit, deltaPosInit) * poidsD * 1.0e20;
              cur_tens_ref_position_tensor += tensor(deltaPosCour, deltaPosInit) * poidsD * 1.0e20;            
            }

            //slip vector...
            deltaCourInit = deltaPosCour - deltaPosInit;

            if((norm2(deltaCourInit) >= m_rrDef*m_rrDef/100.) & (rrInit2 <= m_rrDef*m_rrDef)) {

              slip_vector += deltaCourInit;
              slipped_neighbor_nbr++;
            }
          }  

          if(compute_cinematic_measure) {
              
            //L...
            if(rrCour2 <= 1.2*m_rrDef*m_rrDef) {
                             //v-(from the buffer)  
              deltaVit.x = tab.ext.vx[i] - cells[tab.cell].field_pointer_or_null(field::vx)[tab.part];
              deltaVit.y = tab.ext.vy[i] - cells[tab.cell].field_pointer_or_null(field::vy)[tab.part];
              deltaVit.z = tab.ext.vz[i] - cells[tab.cell].field_pointer_or_null(field::vz)[tab.part];

              cur_tens_cur_position_tensor += tensor(deltaPosCour, deltaPosCour) * poidsD * 1.0e20;
              speed_tens_cur_position_tensor += tensor(deltaVit, deltaPosCour) * poidsD * 1.0e20;            

            }  
          }
        }
   
        //---------- Measures Computation
        if(compute_static_measure) {
          
          //first F
          inv_ref_tens_ref_position_tensor = inverse(ref_tens_ref_position_tensor);
            
          transformation_gradient_tensor = AikBkj(cur_tens_ref_position_tensor, inv_ref_tens_ref_position_tensor);

          // test on nan values
          save_nan(transformation_gradient_tensor); 
          
          transformation_gradient_tensor_data[tab.cell][tab.part] = transformation_gradient_tensor;

          //then the derived... 
          green_lagrange_tensor = 0.5*(transformation_gradient_tensor*transpose(transformation_gradient_tensor) - I);

          green_lagrange_tensor_data[tab.cell][tab.part] = green_lagrange_tensor;
          
          //polar decomposition
          mat3d_polar_decomposition(transformation_gradient_tensor, rotation_tensor, pure_deformation_tensor);
          rotation_tensor_data[tab.cell][tab.part] = rotation_tensor;
          pure_deformation_tensor_data[tab.cell][tab.part] = pure_deformation_tensor;


          //microrotation computation
          rotation_tensor = 0.5*(rotation_tensor - transpose(rotation_tensor)); //rotation_tensor becomes skew(rotation_tensor)

          microrotation_vector.x = 0.5*(rotation_tensor.m32-rotation_tensor.m23);
          microrotation_vector.y = 0.5*(rotation_tensor.m13-rotation_tensor.m31);
          microrotation_vector.z = 0.5*(rotation_tensor.m21-rotation_tensor.m12); 

          microrotation_vector_data[tab.cell][tab.part] = microrotation_vector;     


          if(slipped_neighbor_nbr > 0) {slip_vector = slip_vector/double(slipped_neighbor_nbr);} 
          slip_vector_data[tab.cell][tab.part] = slip_vector;


          //slip orientation computation
          transformation_gradient_tensor = transformation_gradient_tensor - I;
  
          l_constructor_tensor = transformation_gradient_tensor*transpose(transformation_gradient_tensor);
          n_constructor_tensor = transpose(transformation_gradient_tensor)*transformation_gradient_tensor;

	      square_l = l2_norm(l_constructor_tensor);

          if((l_constructor_tensor.m11 > l_constructor_tensor.m22) && (l_constructor_tensor.m11 > l_constructor_tensor.m33)) {
            l_vector.x = sqrt(l_constructor_tensor.m11);
            l_vector.y = sign(l_constructor_tensor.m12)*sqrt(l_constructor_tensor.m22);
            l_vector.z = sign(l_constructor_tensor.m13)*sqrt(l_constructor_tensor.m33);
          }
          
          else if((l_constructor_tensor.m22 > l_constructor_tensor.m11) && (l_constructor_tensor.m22 > l_constructor_tensor.m33)) {
            l_vector.x = sign(l_constructor_tensor.m12)*sqrt(l_constructor_tensor.m11);
            l_vector.y = sqrt(l_constructor_tensor.m22);
            l_vector.z = sign(l_constructor_tensor.m23)*sqrt(l_constructor_tensor.m33);
          }

          else {
            l_vector.x = sign(l_constructor_tensor.m13)*sqrt(l_constructor_tensor.m11);
            l_vector.y = sign(l_constructor_tensor.m23)*sqrt(l_constructor_tensor.m22);
            l_vector.z = sqrt(l_constructor_tensor.m33);
          }

          if(abs(l_vector.z) > 0.0001) {l_vector*sign(l_vector.z);}
          else if(abs(l_vector.y) > 0.0001) {l_vector*sign(l_vector.y);}

          if((n_constructor_tensor.m11 > n_constructor_tensor.m22) && (n_constructor_tensor.m11 > n_constructor_tensor.m33)) {
            n_vector.x = sqrt(n_constructor_tensor.m11);
            n_vector.y = sign(n_constructor_tensor.m12)*sqrt(n_constructor_tensor.m22);
            n_vector.z = sign(n_constructor_tensor.m13)*sqrt(n_constructor_tensor.m33);
          }
         
          else if((n_constructor_tensor.m22 > n_constructor_tensor.m11) && (n_constructor_tensor.m22 > n_constructor_tensor.m33)) {
            n_vector.x = sign(n_constructor_tensor.m12)*sqrt(n_constructor_tensor.m11);
            n_vector.y = sqrt(n_constructor_tensor.m22);
            n_vector.z = sign(n_constructor_tensor.m23)*sqrt(n_constructor_tensor.m33);
          }

          else {
            n_vector.x = sign(n_constructor_tensor.m13)*sqrt(n_constructor_tensor.m11);
            n_vector.y = sign(n_constructor_tensor.m23)*sqrt(n_constructor_tensor.m22);
            n_vector.z = sqrt(n_constructor_tensor.m33);
          }
           
          //forcer l'orientation cohérente entre les vecteurs        
          if(abs(n_vector.z) > 0.0001) {n_vector*sign(n_vector.z);}
          else if(abs(n_vector.y) > 0.0001) {n_vector*sign(n_vector.y);}
            

          m_vector = cross(n_vector, l_vector);
          //m_vector = m_vector*norm(l_vector)/norm(m_vector);
          
    	  l_vector_data[tab.cell][tab.part] = l_vector/norm(l_vector);
  	      m_vector_data[tab.cell][tab.part] = m_vector/norm(m_vector);
	      n_vector_data[tab.cell][tab.part] = n_vector/norm(n_vector);
        }


        if(compute_cinematic_measure) {

          inv_cur_tens_cur_position_tensor = inverse(cur_tens_cur_position_tensor);
        
          speed_gradient_tensor = AikBkj(speed_tens_cur_position_tensor, inv_cur_tens_cur_position_tensor);
 
          save_nan(speed_gradient_tensor);

          speed_gradient_tensor_data[tab.cell][tab.part] = speed_gradient_tensor;

          speed_gradient_tensor  = 0.5*(speed_gradient_tensor - transpose(speed_gradient_tensor));

          vortex_vector.x = 0.5*(speed_gradient_tensor.m32-speed_gradient_tensor.m23);
          vortex_vector.y = 0.5*(speed_gradient_tensor.m13-speed_gradient_tensor.m31);
          vortex_vector.z = 0.5*(speed_gradient_tensor.m21-speed_gradient_tensor.m12); 
  
          vortex_vector_data[tab.cell][tab.part] = vortex_vector;
        }

      }
    };


    //Gradient of a vector from reference system
    template< typename GridT, typename SecondOrderAnalyzerStruct >
    struct RefGradientComputeOp
    {
      using CellsT = decltype( GridT{}->cells() );
      const CellsT m_cells_t0;
      
      const std::vector< std::vector<Vec3d> > & vector_to_grad_data;

      std::vector< std::vector<Mat3d> > & vector_gradient_tensor_data;

      const Mat3d m_xform_t0; // <<<<<<<<<<<<<<<<<<<<<<================================
      const Mat3d m_xform; // <<<<<<<<<<<<<<<<<<<<<<================================
      const Mat3d m_lattice;
      double m_rrDef;

      double (*weight_function)(const double&, const double&) = calc_weight;

      SecondOrderAnalyzerStruct SecondOrderAnalyzer = false;

      template<class ComputeBufferT, class CellParticlesT, class GridParticleLocksT, class ParticleLockT>
      inline void operator ()
        (
        // number of neighbors
        size_t n,

        // buffer holding attached data to particle's neighbors
        const ComputeBufferT& tab,
                
        // data and locks accessors for neighbors
        CellParticlesT* cells,
        GridParticleLocksT locks,
        ParticleLockT & particle_lock
        )
      {
	    Mat3d vector_gradient_tensor;

	    //Gradient constructors
	    Vec3d deltaVec;
	    size_t c_nbh;
	    size_t p_nbh;
 
        Vec3d deltaPosInit;
        Vec3d deltaPosInitCris;

	    Mat3d vec_tens_ref_position_tensor;
	    Mat3d ref_tens_ref_position_tensor;
	    Mat3d inv_ref_tens_ref_position_tensor;

        //Initial position
        Vec3d dtr;

        Mat3d hh0 = m_xform_t0 * m_lattice;

	    Vec3d tmp;
	    tmp.x = m_cells_t0[tab.cell][field::rx][tab.part];
	    tmp.y = m_cells_t0[tab.cell][field::ry][tab.part];
	    tmp.z = m_cells_t0[tab.cell][field::rz][tab.part];

	    Vec3d PosInit = m_xform_t0 * tmp;

	    Vec3d PosInitVois;

	    Mat3d I = make_identity_matrix();

	    // loop over neighborhood
        for(size_t i=0;i<n;i++)
        {

          tmp.x = tab.ext.rx0[i];
          tmp.y = tab.ext.ry0[i];
          tmp.z = tab.ext.rz0[i];

	      PosInitVois = m_xform_t0 * tmp;

	      deltaPosInit = PosInitVois - PosInit;

          dtr.x = dtr.y = dtr.z = 0.0;           
          deal_with_periodicity_ref_pos(deltaPosInit, hh0, dtr);

          double rrInit2 = norm2(deltaPosInit);

          if(rrInit2 <= 1.2*m_rrDef*m_rrDef) {
            tab.nbh.get(i, c_nbh, p_nbh);
	        deltaVec = vector_to_grad_data[c_nbh][p_nbh] - vector_to_grad_data[tab.cell][tab.part];
          }

          double poidsD;

          poidsD = calc_weight(sqrt(rrInit2), m_rrDef);

          if(rrInit2 <= 1.2*m_rrDef*m_rrDef) {

            ref_tens_ref_position_tensor += tensor(deltaPosInit, deltaPosInit) * poidsD * 1.0e20;
            vec_tens_ref_position_tensor += tensor(deltaVec, deltaPosInit) * poidsD * 1.0e20;            
          }       
        }
 
        inv_ref_tens_ref_position_tensor = inverse(ref_tens_ref_position_tensor);
        
        vector_gradient_tensor = AikBkj(vec_tens_ref_position_tensor, inv_ref_tens_ref_position_tensor);

        save_nan(vector_gradient_tensor);

        vector_gradient_tensor_data[tab.cell][tab.part] = vector_gradient_tensor;

        SecondOrderAnalyzer(vector_gradient_tensor, tab.cell, tab.part);
      }
    };
}
