#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_theta_update(int m,
                           double a_sigma2_theta,
                           double b_sigma2_theta,
                           arma::vec theta,
                           double rho_theta_old,
                           arma::mat theta_corr_inv){

double a_sigma2_theta_update = 0.50*m + 
                               a_sigma2_theta;

double b_sigma2_theta_update = 0.50*(1.00 - pow(rho_theta_old, 2.00))*dot(theta, ((theta_corr_inv)*theta)) + 
                               b_sigma2_theta;

double sigma2_theta = 1.00/R::rgamma(a_sigma2_theta_update,
                                     (1.00/b_sigma2_theta_update));

return(sigma2_theta);

}





