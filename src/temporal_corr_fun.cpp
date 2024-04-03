#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List temporal_corr_fun(int m,
                             double rho_theta){

double log_deter_corr_inv = 0.00; 
double sign = 0.00;     
arma::mat corr_inv(m, m); corr_inv.fill(0.00);
corr_inv.diag().fill(1.00 + pow(rho_theta, 2));
corr_inv(0,0) = 1.00;
corr_inv((m - 1), (m - 1)) = 1.00;
for(int j = 0; j < (m - 1); ++j){
  
   corr_inv(j, (j + 1)) = -rho_theta;
   corr_inv((j + 1), j) = -rho_theta;
  
   }
corr_inv = corr_inv/(1.00 - pow(rho_theta, 2));
  
log_det(log_deter_corr_inv, sign, corr_inv);

return Rcpp::List::create(Rcpp::Named("temporal_corr_inv") = corr_inv,
                          Rcpp::Named("log_deter_inv") = log_deter_corr_inv);

}

