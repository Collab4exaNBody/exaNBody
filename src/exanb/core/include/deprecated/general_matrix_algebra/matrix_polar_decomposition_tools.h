#pragma once

////////////////////////////////////////////////////////////////////////////////
//                        POLAR DECOMPOSITION TOOLS
////////////////////////////////////////////////////////////////////////////////


#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LDLT_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition.h"


namespace exanb {

  inline matrix matrix::compute_B_matrix() const {

    assert((M == 3) & (N == 3));

    matrix result(4);

    //construite à la main pour limiter les constructions temporaires et aller plus vite...
    double content[16] = {trace(), matrix_[5]-matrix_[7], matrix_[6]-matrix_[2], matrix_[1]-matrix_[3], matrix_[5]-matrix_[7],matrix_[0]-matrix_[4]-matrix_[8],matrix_[1]+matrix_[3],matrix_[2]+matrix_[6], matrix_[6]-matrix_[2],matrix_[1]+matrix_[3],matrix_[4]-matrix_[0]-matrix_[8],matrix_[5]+matrix_[7],matrix_[1]-matrix_[3],matrix_[2]+matrix_[6],matrix_[5]+matrix_[7],matrix_[8]-matrix_[0]-matrix_[4]};

    result = content;

    return result;  
  };


  inline matrix vector_::compute_Q_matrix() const {

    assert(M == 4);

    matrix result(3);

    //construite à la main pour limiter les constructions temporaires et aller plus vite dans l'implémentation...
    double content[9] = {1-2*(vecteur[2]*vecteur[2]+vecteur[3]*vecteur[3]), 2*(vecteur[1]*vecteur[2]+vecteur[0]*vecteur[3]), 2*(vecteur[1]*vecteur[3]-vecteur[0]*vecteur[2]), 2*(vecteur[1]*vecteur[2]-vecteur[0]*vecteur[3]), 1-2*(vecteur[1]*vecteur[1]+vecteur[3]*vecteur[3]), 2*(vecteur[2]*vecteur[3]+vecteur[0]*vecteur[1]), 2*(vecteur[1]*vecteur[3]+vecteur[0]*vecteur[2]), 2*(vecteur[2]*vecteur[3]-vecteur[0]*vecteur[1]), 1-2*(vecteur[1]*vecteur[1]+vecteur[2]*vecteur[2])};

    result = content;

    return result;  
  };


  inline double Horners_polynomial_identification(const double Polynom[], size_t Polynom_degree, double doubl_) { //Algorithme de Hörner pour l'évaluation d'un polynôme

    double result = Polynom[Polynom_degree];
    
    for(int i=Polynom_degree-1; i>-1; i--) {
        result = result*doubl_+Polynom[i]; 
    }
    
    return result;
  };


  //==================================================================== Lemma 2.2
  inline void matrix::compute_forth_order_pseudo_characteristic_polynoms(double Polynom[], double Polynom_derivative[]) const { 
    //que dans le cadre de l'article

    double matrix_square_Frobeniux_norm = square_Frobenius_norm();

    Polynom[0] = matrix_square_Frobeniux_norm*matrix_square_Frobeniux_norm - 4*(adjugate().square_Frobenius_norm());
    Polynom[1] = -8*det();
    Polynom[2] = -2*matrix_square_Frobeniux_norm;
    Polynom[3] = 0;
    Polynom[4] = 1;

    Polynom_derivative[0] = Polynom[1];
    Polynom_derivative[1] = 2*Polynom[2];
    Polynom_derivative[2] = 0;
    Polynom_derivative[3] = 4*Polynom[4];
  }; 






  //=============================================================== Algorithme 3.4
  inline double matrix::Newton_dominant_eigen_value(double b, bool debug_mod) const {
    //Newton's method to estimate dominant eigenvalue

    double x, x_old;
    double p_0, p_1;

    if(debug_mod) 
      std::cout << "calcul des valeurs propres par méthode de Newton" <<  std::endl;
    
    x = sqrt(3.);
    x_old = 3.;

    double B_characteristic_polynomial[5];
    double B_characteristic_polynomial_derivative[4];
    
    compute_forth_order_pseudo_characteristic_polynoms(B_characteristic_polynomial, B_characteristic_polynomial_derivative);

    while(abs(x_old-x) > 1e-15) {//condition de convergence de x
      
      x_old = x;
      
      p_0 = Horners_polynomial_identification(B_characteristic_polynomial, 4, x);
      p_1 = Horners_polynomial_identification(B_characteristic_polynomial_derivative, 3, x);
      
      x = x-p_0/p_1;
    }

    return x;
  };


  //=============================================================== Algorithme 3.3
  inline double matrix::compute_dominant_eigen_value(double b, double d, bool debug_mod) const {

    double lambda_1;
  

    if(b+1/3. > 1e-4) {

      if(debug_mod) 
        std::cout << "det B plus grand -1/3" << std::endl;

      double c, delta_0, delta_1, alpha, z, s;

      c = 8*d;
      delta_0 = 1+3*b;
      delta_1 = -1. + 27/16.*c*c + 9*b;
      alpha = delta_1/pow(delta_0, 3/2.);

      if((alpha > -1) && (alpha < 1)) 
        z = 4./3.*(1+sqrt(delta_0)*cos(acos(alpha)/3.));//acos(-1) return(nan)   
      else  
        z = 4./3.*(1+sqrt(delta_0)*cos(-(alpha-1)*M_PI/6.));

      s =  sqrt(z)/2.;
      if(4-z+c/s > 0.) 
        lambda_1 = s+sqrt(4-z+c/s)/2.;
      else 
        lambda_1 = s;
    }

    else 
      lambda_1 = Newton_dominant_eigen_value(b);

    return lambda_1;
  };



  //algorithme inverse pour la recherche de vecteur propre
  inline void matrix::inverse_iteration_step(vector_& v, double& lambda) const {

    assert((M==N) & (v.M==M));

    v = (matrice()-lambda*identity_(M)).inverse()*v;

    v = v/v.norm();

    lambda = (v.transpose()*matrice()*v)(0);
  };



    //Algorithme d'orthonormalisation de Grahm-Schmidt
  inline void matrix::grahm_schmidt_algorithm() {

    vector_ col_n(M);
    vector_ orthonormalized_column[N]; 

    for(size_t n=0; n<N; n++) {

      col_n = column_extractor(n);
      
      if(n == 0) orthonormalized_column[0] = col_n/col_n.norm();

      else {

        orthonormalized_column[n] = col_n;

        for(size_t j=0; j<n; j++) orthonormalized_column[n] -= (col_n*orthonormalized_column[j])*orthonormalized_column[j];

        orthonormalized_column[n] = orthonormalized_column[n]/orthonormalized_column[n].norm();
      }

      column_adder(orthonormalized_column[n], n);
    }    
  };
}
