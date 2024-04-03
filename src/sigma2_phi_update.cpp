#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_phi_update(int n_grid,
                         double a_sigma2_phi,
                         double b_sigma2_phi,
                         arma::vec phi_star,
                         arma::mat phi_star_corr_inv){

double a_sigma2_phi_update = 0.50*n_grid + 
                             a_sigma2_phi;

double b_sigma2_phi_update = 0.50*dot(phi_star, ((phi_star_corr_inv)*phi_star)) + 
                             b_sigma2_phi;

double sigma2_phi = 1.00/R::rgamma(a_sigma2_phi_update,
                                   (1.00/b_sigma2_phi_update));

return(sigma2_phi);

}





