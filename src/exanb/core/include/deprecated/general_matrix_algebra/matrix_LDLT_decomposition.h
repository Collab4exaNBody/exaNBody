#pragma once

////////////////////////////////////////////////////////////////////////////////
//                            DÉCOMPOSTIONS LDLT
////////////////////////////////////////////////////////////////////////////////


#include "matrix.h"
#include "matrix_object.h"
#include "vector.h"
#include "matrix_operators.h"
#include "matrix_determinant_trace_norme.h"
#include "matrix_pivoting.h"
#include "matrix_inverse_transpose_adjugate.h"
#include "matrix_3x3_eigen_decomposition.h"
#include "matrix_LU_decomposition.h"
#include "matrix_polar_decomposition_tools.h"
#include "matrix_polar_decomposition.h"

using namespace std;

namespace exanb {


  ////////////////////////////////////////////////////////////////////////////////
  //                   DÉCOMPOSTIONS LDLT AVEC PIVOT DIAGONAL
  ////////////////////////////////////////////////////////////////////////////////

  inline void matrix::diagonal_PLDLTPT_decomposition(matrix& P, matrix& L, matrix& D) const {

    size_t i,j,m,n,max_line;
    double max;
  

    //initialisation de D comme copie de A
    D = matrice();

    //Initialisations de L et P
    L = matrix_zeros(M);
    P = identity_(M);

    matrix pivot_matrix(M); //matrice support des transformations élémentaires

    for(n=0; n<M-1; n++) {
      
      max=0;
      max_line=n;

      for(m=n; m<M; m++) {
    
        if(abs(D.matrix_[m+m*N]) > abs(max)) {

          max = D.matrix_[m+m*N];
          max_line = m;
        }
      }

      if (max_line != n) {

        pivot_matrix = line_inversion_matrix(max_line, n, M);

        D = pivot_matrix*D*pivot_matrix;
        L = pivot_matrix*L;
        P*=pivot_matrix;            
      }

      pivot_matrix = D.matrix_to_zero_columns(n, "LUdecomposition");

      L -= (pivot_matrix-identity_(M));   

      D = pivot_matrix*D; //une seule matrice à mettre à jour
        
    }

    for(i=0; i<M-1; i++) {
      for(j=i+1; j<N; j++) D.matrix_[j+i*M] = 0;
    }
    
    L = L+identity_(M);    
  };





  ////////////////////////////////////////////////////////////////////////////////
  //                   DÉCOMPOSTIONS LDLT PAR BLOCK (Bunch Parlett pivoting)
  ////////////////////////////////////////////////////////////////////////////////

  inline void matrix::block_LDLT_decompostion(matrix& L, matrix& D) const {
    
    double mu_0,mu_1,mu_cur, nu,nu_cur;
    size_t i,j,n;
    double alpha = 0.6403882032022076;

    D = matrice();
    L = identity_(M);//matrix_zeros(M);
    matrix P(2);//,  pivot_matrix(M), pivot_matrix2(M);
    //matrix C; //matrices de construction
    
    for(n=0; n<M-2; n++) {
    
      mu_0 = 0;
      mu_1 = 0; 

      nu = 0;

      
      for(i=n; i<M; i++) {    
        for(j=i; j<M; j++) {

          mu_cur = abs(D.matrix_[j+i*M]);
          nu_cur = abs(D.matrix_[j+j*M]*D.matrix_[i+i*M]-D.matrix_[j+i*M]*D.matrix_[j+i*M]);

          if(mu_cur > mu_0) mu_0 = mu_cur;

          if((i == j) & (mu_cur > mu_1)) mu_1 = mu_cur;

          if(nu_cur > nu) nu = nu_cur;            
        }
      }

            
      if(mu_1 >= alpha*mu_0) { // 1x1 pivoting
        
        assert(M-n-1 > 0);
        matrix C(M-n-1,1);
        C = matrix_zeros(M-n-1,1);

        for(i=0; i<M-n-1; i++) C.matrix_[i] = D.matrix_[n+(i+n+1)*M];

        matrix pivot_matrix(M-n-1);
        pivot_matrix = C*C.transpose();

        matrix pivot_matrix2(M);
        pivot_matrix2 = matrix_zeros(M);

        for(i=0; i<M; i++){
          for(j=0; j<M; j++) {
        
            if((i<n) | (j<n) | ((i==n) & (j==n))) pivot_matrix2.matrix_[j+i*M] = 0;

            else if(i==n) pivot_matrix2.matrix_[j+i*M] = C.matrix_[j-n-1];

            else if(j==n) {
              pivot_matrix2.matrix_[j+i*M] = C.matrix_[i-n-1];
              L.matrix_[j+i*M] = C.matrix_[i-n-1]/D.matrix_[n+n*M];
            }

            else pivot_matrix2.matrix_[j+i*M] = pivot_matrix.matrix_[(j-n-1)+(i-n-1)*pivot_matrix.N]/D.matrix_[n+n*M];
          }
        } 

        D -= pivot_matrix2;
      }
      
      
      else {
        matrix C(M-n-2, 2);
        C = matrix_zeros(M-n-2,2);
        P = matrix_zeros(2);

        for(i=0; i<M-n-2; i++) {
          C.matrix_[i*2] = D.matrix_[n+(i+n+2)*M];
          C.matrix_[1+i*2] = D.matrix_[n+1+(i+n+2)*M];
        }

        P.matrix_[0] = D.matrix_[n+n*N];
        P.matrix_[1] = D.matrix_[n+1+n*N];
        P.matrix_[2] = D.matrix_[n+(n+1)*N];
        P.matrix_[3] = D.matrix_[n+1+(n+1)*N];
        
        matrix pivot_matrix(M-n-2);
        pivot_matrix = C*P.inverse()*C.transpose();
        
        matrix P2(M-n-2,1);
        P2 = C*P.inverse();
        
        matrix pivot_matrix2(M);
        pivot_matrix2 = matrix_zeros(M);

        for(i=0; i<M; i++){
          for(j=0; j<M; j++) {

            if((i<n) | (j<n) | ((i==n) & (j==n)) | ((i==n+1) & (j==n+1))| ((i==n) & (j==n+1))| ((i==n+1) & (j==n)));

            else if(i==n) pivot_matrix2.matrix_[j+i*M] = C.matrix_[(j-n-2)*2];

            else if(i==n+1) pivot_matrix2.matrix_[j+i*M] = C.matrix_[1+(j-n-2)*2];

            else if(j==n) {
                pivot_matrix2.matrix_[j+i*M] = C.matrix_[(i-n-2)*2];
                L.matrix_[j+i*M] = P2.matrix_[(i-n-2)*2];  
            }

            else if(j==n+1) {
                pivot_matrix2.matrix_[j+i*M] = C.matrix_[1+(i-n-2)*2];
                L.matrix_[j+i*M] = P.matrix_[1+(i-n-2)*2];  
            }

            else pivot_matrix2.matrix_[j+i*M] = pivot_matrix.matrix_[(j-n-2)+(i-n-2)*pivot_matrix.N];
          }
        } 

        D -= pivot_matrix2;
      }
    }
  };
};
