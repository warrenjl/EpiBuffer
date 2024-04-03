#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List spatial_corr_fun(double phi,
                            arma::mat spatial_dists){

double log_deter_inv = 0.00; 
double sign = 0.00;     

arma::mat corr = exp(-phi*spatial_dists);
arma::mat corr_inv = inv_sympd(corr);
log_det(log_deter_inv, sign, corr_inv);

return Rcpp::List::create(Rcpp::Named("spatial_corr_inv") = corr_inv,
                          Rcpp::Named("log_deter_inv") = log_deter_inv);

}
