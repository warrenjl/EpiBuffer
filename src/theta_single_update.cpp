#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List theta_single_update(arma::mat x, 
                               arma::vec off_set,
                               int n_ind,
                               int m,
                               arma::vec omega,
                               arma::vec lambda,
                               arma::vec beta,
                               double sigma2_theta_old,
                               double rho_theta_old,
                               arma::mat theta_corr_inv,
                               arma::uvec radius_pointer_uvec_old, 
                               arma::mat Z){

arma::mat omega_mat(n_ind, m);
for(int j = 0; j < m; ++j){
   omega_mat.col(j) = omega;
   }
  
arma::mat Z_trans = trans(Z);

arma::mat cov_theta = inv_sympd(Z_trans*(omega_mat%Z) + 
                                ((1.00 - pow(rho_theta_old, 2.00))/sigma2_theta_old)*theta_corr_inv);

arma::vec mean_theta = cov_theta*(Z_trans*(omega%(lambda - off_set - x*beta)));

arma::mat ind_norms = arma::randn(1, m);
arma::vec theta = mean_theta + 
                  trans(ind_norms*arma::chol(cov_theta));
arma::vec theta_keep = theta.elem(radius_pointer_uvec_old - 1);

return Rcpp::List::create(Rcpp::Named("theta") = theta,
                          Rcpp::Named("theta_keep") = theta_keep);

}






