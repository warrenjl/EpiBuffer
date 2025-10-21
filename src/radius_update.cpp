#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List radius_update(arma::vec radius_range,
                         int exposure_definition_indicator,
                         arma::mat v_exposure_dists,
                         int p_q,
                         int n_ind,
                         int m,
                         int m_max,
                         arma::mat x,
                         arma::mat q,
                         arma::vec off_set,
                         arma::vec omega,
                         arma::vec lambda,
                         arma::vec beta,
                         arma::vec eta,
                         double radius_old,
                         double radius_trans_old,
                         arma::vec exposure,
                         arma::mat Z,
                         double metrop_var_radius,
                         int acctot_radius){

/*Second*/
arma::vec exposure_old = exposure;
arma::mat Z_old = Z;

double denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) + 
               -radius_trans_old +
               -2.00*log(1.00 + exp(-radius_trans_old));

/*First*/
double radius_trans = R::rnorm(radius_trans_old, 
                               sqrt(metrop_var_radius));
double radius = (radius_range(1)*exp(radius_trans) + radius_range(0))/(exp(radius_trans) + 1.00);
arma::mat radius_mat(n_ind, m); radius_mat.fill(radius);

//Cumulative Counts
if(exposure_definition_indicator == 0){
  
  arma::umat comparison = (v_exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::sum(numeric_mat,
                       1);
  exposure = exposure/m_max;
  
  }

//Spherical
if(exposure_definition_indicator == 1){
  
  arma::mat fast = v_exposure_dists/radius_mat;
  arma::mat corrs = 1.00 +
                    -1.50*fast +
                    0.50*pow(fast, 3);
  arma::umat comparison = (v_exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  arma::mat prod = corrs%numeric_mat;
  exposure = arma::sum(prod,
                       1);
  exposure = exposure/m_max;
  
  }

//Presence/Absence
if(exposure_definition_indicator == 2){
  
  arma::umat comparison = (v_exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::max(numeric_mat,
                       1);
  
  }

for(int j = 0; j < p_q; ++j){
   Z.col(j) = exposure%q.col(j);
   }

double numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) + 
               -radius_trans +
               -2.00*log(1.00 + exp(-radius_trans));
  
/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  radius_trans = radius_trans_old;
  radius = radius_old;
  exposure = exposure_old;
  Z = Z_old;
  acc = 0;
  
  }
acctot_radius = acctot_radius + 
                acc;

return Rcpp::List::create(Rcpp::Named("radius") = radius,
                          Rcpp::Named("acctot_radius") = acctot_radius,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);

}



