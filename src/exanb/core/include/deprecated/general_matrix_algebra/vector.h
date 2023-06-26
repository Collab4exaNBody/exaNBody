#pragma once

#include "matrix.h"

using namespace std;



namespace exanb {

  struct vector_ {

    double* vecteur; //vecteur 1D

    int M; //dimension de la matrice 

    inline vector_();
    inline vector_(int n_lin);
    inline vector_(int n_lin, double double_[]);
    inline ~vector_();

    inline void plot() const;

    inline double& x();
    inline double& y();
    inline double& z();

    inline double& operator()(int m);
    inline double operator()(int m) const;

    inline void operator=(double double_[]);
    inline void operator=(const vector_& vec);

    //produit par un scalaire
    inline vector_ operator*(double scal) const;
/*        //produit terme à terme*/
/*        vector_ operator*(const vector_& vec) const;*/
    //produit scalaire
    inline double operator*(const vector_& vec) const;
    //produit tensoriel
    inline matrix operator^(const vector_& vec) const;
    //produit matriciel avec un vecteur ligne
    inline matrix operator*(const matrix& Mat) const;
    inline vector_ fake_product(const vector_& vec) const;

    inline vector_ operator+(const vector_& vec) const;
    inline vector_ operator-(const vector_& vec) const;

    inline matrix transpose() const;

    inline double norm() const;

    inline matrix compute_Q_matrix() const;
  };

  //Constructeur et Destructeur
  inline vector_::vector_() {
    M=-1;
  };


  inline vector_::vector_(int n_lin) {

    M = n_lin;
    vecteur = new double[M];
  };


  inline vector_::vector_(int n_lin, double double_[]) {

    int i;
    M = n_lin;
    vecteur = new double[M];

    for(i=0; i<n_lin; i++) vecteur[i] = double_[i];
  };


  inline vector_::~vector_() {
    if (M!=-1) delete[] vecteur;
  };


  inline vector_ unity(int n_lin) {

    int i;
    double unity_line[n_lin];

    for(i=0; i<n_lin; i++) unity_line[i] = 1;

    return vector_(n_lin, unity_line);
  };

  inline vector_ zeros(int n_lin) {

    int i;
    double unity_line[n_lin];

    for(i=0; i<n_lin; i++) unity_line[i] = 0;

    return vector_(n_lin, unity_line);
  };



  inline void vector_::plot() const {

    int m;

    for(m=0; m<M; m++) {

      cout << " [" << vecteur[m] << ']' << endl;
    }  
    cout << endl;
  };


  inline double& vector_::x() {
    return vecteur[0];
  }


  inline double& vector_::y() {
    return vecteur[1];
  }


  inline double& vector_::z() {
    return vecteur[2];
  }


  inline double& vector_::operator()(int m) {
    assert(m<M); 
    return vecteur[m];
  };


  inline double vector_::operator()(int m) const {
    assert(m<M); 
    return vecteur[m];
  };


  inline void vector_::operator=(double double_[]) {

    int m;
    for(m=0; m<M; m++) vecteur[m] = double_[m];
  };


  inline void vector_::operator=(const vector_& vec) {
    
    int m;

    if(M!=-1) {
      delete[] vecteur;
    }

    M=vec.M;

    if(M!=-1) {

      vecteur = new double[M];
      for(m=0; m<M; m++) vecteur[m] = vec.vecteur[m];
    }
  };


  inline vector_ vector_::operator*(double scal) const { //produit par un scalaire

    int m;
    vector_ result(M);
    
    for(m=0; m<M; m++) result.vecteur[m] = vecteur[m]*scal;

    return result;
  };


  inline vector_ operator*(double scal, const vector_& vec) {
    return vec*scal;
  };


  inline vector_ operator/(const vector_& vec, double scal) {
    return vec*(1/scal);
  };


  inline void operator/=(vector_& vec, double scal) {
    vec = vec*(1/scal);
  };


//vector_ vector_::operator*(const vector_& vec) const { //produit terme à terme

//    assert(M==vec.M);

//    vector_ result(M);

//    for(int m=0; m<M; m++) {

//        result.vecteur[m] = vecteur[m]* vec.vecteur[m];
//    }

//    return result;
//};


  inline double vector_::operator*(const vector_& vec) const { //produit scalaire

    assert(M==vec.M);

    int m;
    double result = 0;

    for(m=0; m<M; m++) result += vecteur[m]*vec.vecteur[m];

    return result;
  };


  inline vector_ vector_::fake_product(const vector_& vec) const {

    assert(M==vec.M);

    int m;
    vector_ result(M);

    for(m=0; m<M; m++) result.vecteur[m] = vecteur[m]*vec.vecteur[m];

    return result;
  };


  inline matrix vector_::operator^(const vector_& vec) const { //produit matriciel (seulement avec un vecteur ligne)

    assert(vec.M==M);

    int m,n;
    matrix result(M);

    for(m=0; m<M; m++) {
      for(n=0; n<M; n++) result.matrix_[n+m*M] = vecteur[m]*vec.vecteur[n];
    }    

    return result;
  };


  inline matrix vector_::operator*(const matrix& Mat) const { //produit matriciel (seulement avec un vecteur ligne)

    assert((Mat.M==1) & (M==Mat.N));

    int m,n;
    matrix result(M);

    for(m=0; m<M; m++) {
      for(n=0; n<M; n++) result.matrix_[n+m*M] = vecteur[m]*Mat.matrix_[n];
    }    

    return result;
  };



  inline vector_ vector_::operator+(const vector_& vec) const {

    assert(M==vec.M);

    int m;
    vector_ result(M);
    
    for(m=0; m<M; m++) result.vecteur[m] = vecteur[m] + vec.vecteur[m];

    return result;
  };


  inline void operator+=(vector_& vec1, const vector_& vec2) {

    vec1 = vec1+vec2;
  };


  inline vector_ vector_::operator-(const vector_& vec) const {

    assert(M==vec.M);

    int m;
    vector_ result(M);
    
    for(m=0; m<M; m++) result.vecteur[m] = vecteur[m] - vec.vecteur[m];

    return result;
  };


  inline void operator-=(vector_& vec1, const vector_& vec2) {

    vec1 = vec1-vec2;
  };


  inline matrix vector_::transpose() const {

    int m;
    matrix result(1, M);

    for(m=0; m<M; m++) result.matrix_[m] = vecteur[m];

    return result;
  };


  inline double vector_::norm() const { //norme 2

    int m;
    double norm = 0;
    
    for(m=0; m<M; m++) norm += vecteur[m]*vecteur[m]; 

    return sqrt(norm);  
  };


  inline vector_ vect_product(const vector_& vec1, const vector_& vec2) { //produit vectoriel

    assert(vec1.M == vec2.M);

    vector_ result(vec1.M);

    result.vecteur[0] = vec1.vecteur[1]*vec2.vecteur[2]-vec2.vecteur[1]*vec1.vecteur[2];
    result.vecteur[1] = vec1.vecteur[2]*vec2.vecteur[0]-vec2.vecteur[2]*vec1.vecteur[0];
    result.vecteur[2] = vec1.vecteur[0]*vec2.vecteur[1]-vec2.vecteur[0]*vec1.vecteur[1];

    return result;
  };

}
