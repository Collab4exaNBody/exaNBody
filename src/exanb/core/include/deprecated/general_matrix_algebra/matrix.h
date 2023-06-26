#pragma once

#include <iostream> 
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>


using namespace std;


namespace exanb {

  struct vector_;

  struct matrix {
          
    ////////////////////////////////////////////////////////////////////////
    //                            OBJET MATRICE
    ////////////////////////////////////////////////////////////////////////

    double* matrix_; //matrice = vecteur 1D

    size_t M, N; //dimension de la matrice 

    inline matrix();
    inline matrix(size_t n_lin, size_t n_col);
    inline matrix(size_t n_lin, size_t n_col, double double_[]);
    inline matrix(size_t n_lin);
    inline matrix(size_t n_lin, double double_[]);
    inline ~matrix();

    inline matrix matrice() const;

    inline void plot() const;

    inline double& operator()(size_t m, size_t n); //permet de modifier la variable Mat(m,n)
    inline double operator()(size_t m, size_t n) const;//acceder à la variable Mat(m,n)

    inline void operator=(double double_[]);
    inline void operator=(const matrix& Mat);
    //inline const matrix& operator=(const matrix& Mat);
    
    
    ////////////////////////////////////////////////////////////////////////
    //                          OPÉRATEURS
    ////////////////////////////////////////////////////////////////////////

    //produit par un sclaire
    inline matrix operator*(double scal) const;
    //produit contracté
    inline matrix operator*(const matrix& Mat) const;
    //produit matricielle par un vecteur
    inline vector_ operator*(const vector_& vec) const;

    inline matrix operator+(const matrix& Mat) const;
    inline matrix operator-(const matrix& Mat) const;

    inline double double_contraction_product(const matrix& Mat) const;



    ////////////////////////////////////////////////////////////////////////
    //                         TRANSFORMATIONS
    ////////////////////////////////////////////////////////////////////////

    inline matrix inverse() const;
    inline matrix transpose() const;
    inline matrix adjugate() const;


    ////////////////////////////////////////////////////////////////////////
    //                           "MESURES"
    ////////////////////////////////////////////////////////////////////////

    inline double trace() const;
    inline double det_diag() const;
    inline double det_permut() const;
    inline double det() const;

    inline double max_matrix() const;
    inline double square_Frobenius_norm() const;


    ////////////////////////////////////////////////////////////////////////
    //                          OUTILS pivo
    ////////////////////////////////////////////////////////////////////////

    inline matrix matrix_to_zero_columns(size_t column_index_to_compute, string inversion_or_LUdecomposition) const;

    inline void partial_LU_decomposition(matrix& P, matrix& L, matrix& U) const;
    inline void complete_LU_decomposition(matrix& P_l, matrix& P_c, matrix& L, matrix& U) const;
    inline void diagonal_PLDLTPT_decomposition(matrix& P, matrix& L, matrix& D) const;
    inline void block_LDLT_decompostion(matrix& L, matrix& D) const;


    ////////////////////////////////////////////////////////////////////////
    //                      DÉCOMPOSITION POLAIRE
    ////////////////////////////////////////////////////////////////////////

    inline void polar_decomposition(matrix& Mat_ortho, matrix& Mat_semdef_pos, bool debug_mod=false) const; 
    inline void polar_decomposition_reference(matrix& Mat_ortho, matrix& Mat_semdef_pos, matrix& B, double& d, double norm, bool debug_mod=false) const; 


    ////////////////////////////////////////////////////////////////////////
    //                    OUTILS DÉCOMPOSITION POLAIRE
    ////////////////////////////////////////////////////////////////////////

    inline matrix compute_B_matrix() const;
    inline double compute_dominant_eigen_value(double b, double d, bool debug_mod=false) const; //pour les matrice 4x4
    inline double Newton_dominant_eigen_value(double b, bool debug_mod=false) const;
    inline void compute_forth_order_pseudo_characteristic_polynoms(double Polynom[], double Polynom_derivative[]) const;
    
    inline void inverse_iteration_step(vector_& v, double& lambda) const;
    inline void grahm_schmidt_algorithm();
      

    inline vector_ column_extractor(size_t col_index) const;
    inline void column_adder(const vector_& vec, size_t col_index);
  };


  ////////////////////////////////////////////////////////////////////////
  //                      MATRICES "STANDARDS"
  ////////////////////////////////////////////////////////////////////////

  matrix identity_(size_t M);
  matrix matrix_zeros(size_t M);
  matrix matrix_zeros(size_t M, size_t N);


  ////////////////////////////////////////////////////////////////////////
  //                          OPÉRATEURS
  ////////////////////////////////////////////////////////////////////////

  matrix operator*(double scal, const matrix& Mat);
  matrix operator/(const matrix& Mat, double scal);
  void operator+=(matrix& Mat1, const matrix& Mat2);
  void operator-=(matrix& Mat1, const matrix& Mat2);
  void operator*=(matrix& Mat1, const matrix& Mat2);
  void operator/=(matrix& Mat1, double scal);



  ////////////////////////////////////////////////////////////////////////
  //                          OUTILS pivo
  ////////////////////////////////////////////////////////////////////////

  void matrix_update_front(matrix& Mat1, matrix& Mat2, const matrix& operation_matrix);
  void matrix_update_front(matrix& Mat1, matrix& Mat2, matrix& Mat3, const matrix& operation_matrix);
  void matrix_update_back(matrix& Mat1, matrix& Mat2, const matrix& operation_matrix);
  void matrix_update_back(matrix& Mat1, matrix& Mat2, matrix& Mat3, const matrix& operation_matrix);

  matrix matrix_to_divide_a_line(size_t line_to_divide_index, double divider, size_t M);
  matrix line_inversion_matrix(size_t l1, size_t l2, size_t N);



  ////////////////////////////////////////////////////////////////////////
  //                          "MESURES"
  ////////////////////////////////////////////////////////////////////////

  //surcharge pour le fun car tout pour le fun !
  double trace(const matrix& Mat);
  double det_diag(const matrix& Mat);
  double det_permut(const matrix& Mat);
  double det(const matrix& Mat);



  ////////////////////////////////////////////////////////////////////////
  //                    OUTILS DÉCOMPOSITION POLAIRE
  ////////////////////////////////////////////////////////////////////////

  double Horners_polynomial_identification(const double Polynom[], size_t Polynom_degree, double doubl_);



  ////////////////////////////////////////////////////////////////////////
  //                RÉDUCTION POUR LES MATRICES 3x3
  ////////////////////////////////////////////////////////////////////////

  vector_ compute_eigenvalue(const matrix& A, bool debug_mod=false);
  vector_ compute_eigenvector(const matrix& A, const vector_& eigenvalues, size_t value_index, bool debug_mod=false);
  vector_ compute_eigenvector(const matrix& A, double eigenvalue, bool debug_mod=false);

}
