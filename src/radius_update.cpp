#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List radius_update(arma::vec radius_range,
                         int exposure_definition_indicator,
                         arma::mat exposure_dists,
                         int p_d,
                         int n_ind,
                         int m,
                         int m_max,
                         arma::mat x,
                         arma::vec off_set,
                         arma::vec omega,
                         arma::vec lambda,
                         arma::vec beta,
                         arma::vec eta,
                         double radius_old,
                         arma::vec theta,
                         double radius_trans_old,
                         arma::vec poly,
                         arma::vec exposure,
                         arma::mat Z,
                         double metrop_var_radius,
                         int acctot_radius){

/*Second*/
arma::vec poly_old = poly;
arma::vec exposure_old = exposure;
arma::mat Z_old = Z;
arma::vec theta_old = theta;

double denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) + 
               -radius_trans_old +
               -2.00*log(1.00 + exp(-radius_trans_old));

/*First*/
double radius_trans = R::rnorm(radius_trans_old, 
                               sqrt(metrop_var_radius));
double radius = (radius_range(1)*exp(radius_trans) + radius_range(0))/(exp(radius_trans) + 1.00);
arma::mat radius_mat(n_ind, m); radius_mat.fill(radius);
for(int j = 0; j < (p_d + 1); ++j){
   poly(j) = pow((radius - radius_range(0))/(radius_range(1) - radius_range(0)), j);
   }

//Cumulative Counts
if(exposure_definition_indicator == 0){
  
  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::sum(numeric_mat,
                       1);
  exposure = exposure/m_max;
  
  }

//Spherical
if(exposure_definition_indicator == 1){
  
  arma::mat corrs = 1.00 +
                    -1.50*(exposure_dists/radius_mat) +
                    0.50*pow((exposure_dists/radius_mat), 3);
  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  arma::mat prod = corrs%numeric_mat;
  exposure = arma::sum(prod,
                       1);
  exposure = exposure/m_max;
  
  }

//Presence/Absence
if(exposure_definition_indicator == 2){
  
  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::max(numeric_mat,
                       1);
  
  }

for(int j = 0; j < (p_d + 1); ++j){
   Z.col(j) = exposure*poly(j);
   }
theta = dot(poly, eta);

double numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) + 
               -radius_trans +
               -2.00*log(1.00 + exp(-radius_trans));
  
/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  radius_trans = radius_trans_old;
  radius = radius_old;
  poly = poly_old;
  exposure = exposure_old;
  Z = Z_old;
  theta = theta_old;
  acc = 0;
  
  }
acctot_radius = acctot_radius + 
                acc;

return Rcpp::List::create(Rcpp::Named("radius") = radius,
                          Rcpp::Named("acctot_radius") = acctot_radius,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("poly") = poly,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);

}



