#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List tau_phi_update(int n_grid,
                          double tau_phi_old,
                          arma::vec phi_star,
                          arma::mat phi_star_corr_inv,
                          double metrop_var_tau_phi,
                          int acctot_tau_phi){

/*Second*/
double tau_phi_trans_old = log(tau_phi_old/(1.00 - tau_phi_old));

double denom = -n_grid*log(tau_phi_old) + 
               -0.50*dot(phi_star, (phi_star_corr_inv*phi_star))/pow(tau_phi_old, 2.00) +
               -tau_phi_trans_old +
               -2.00*log(1.00 + exp(-tau_phi_trans_old));

/*First*/
double tau_phi_trans = R::rnorm(tau_phi_trans_old, 
                                sqrt(metrop_var_tau_phi));
double tau_phi = 1.00/(1.00 + exp(-tau_phi_trans));

double numer = -n_grid*log(tau_phi) + 
               -0.50*dot(phi_star, (phi_star_corr_inv*phi_star))/pow(tau_phi, 2.00) +
               -tau_phi_trans +
               -2.00*log(1.00 + exp(-tau_phi_trans));

/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  tau_phi = tau_phi_old;
  acc = 0;
  
  }
acctot_tau_phi = acctot_tau_phi + 
                 acc;

return Rcpp::List::create(Rcpp::Named("tau_phi") = tau_phi,
                          Rcpp::Named("acctot_tau_phi") = acctot_tau_phi);

}



