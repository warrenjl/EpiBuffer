#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec eta_update(arma::mat x,
                     arma::vec off_set,
                     int n_ind,
                     int p_q,
                     double sigma2_eta,
                     arma::vec omega,
                     arma::vec lambda,
                     arma::vec beta,
                     arma::mat Z){

arma::mat omega_mat(n_ind, p_q);
for(int j = 0; j < p_q; ++j){
   omega_mat.col(j) = omega;
   }

arma::mat Z_trans = trans(Z);

arma::mat cov_eta = inv_sympd(Z_trans*(omega_mat%Z) + 
                              (1.00/sigma2_eta)*eye(p_q, p_q));

arma::vec mean_eta = cov_eta*(Z_trans*(omega%(lambda - off_set - x*beta)));

arma::mat ind_norms = arma::randn(1, p_q);
arma::vec eta = mean_eta + 
                trans(ind_norms*arma::chol(cov_eta));

return eta;

}



