#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec beta_update(arma::mat x,
                      arma::vec off_set,
                      int n_ind,
                      int p_x,
                      double sigma2_beta,
                      arma::vec omega,
                      arma::vec lambda,
                      arma::vec eta_old,
                      arma::mat Z){

arma::mat omega_mat(n_ind, p_x);
for(int j = 0; j < p_x; ++j){
   omega_mat.col(j) = omega;
   }

arma::mat x_trans = trans(x);

arma::mat cov_beta = inv_sympd(x_trans*(omega_mat%x) + 
                               (1.00/sigma2_beta)*eye(p_x, p_x));

arma::vec mean_beta = cov_beta*(x_trans*(omega%(lambda - off_set - Z*eta_old)));

arma::mat ind_norms = arma::randn(1, p_x);
arma::vec beta = mean_beta + 
                 trans(ind_norms*arma::chol(cov_beta));

return beta;

}



