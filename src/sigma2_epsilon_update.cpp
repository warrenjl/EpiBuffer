#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double sigma2_epsilon_update(arma::vec y,
                             arma::mat x,
                             arma::vec off_set,
                             int n_ind,
                             double a_sigma2_epsilon,
                             double b_sigma2_epsilon,
                             arma::vec beta_old,
                             arma::vec theta_keep_old,
                             arma::mat Z){

double a_sigma2_epsilon_update = 0.50*n_ind + 
                                 a_sigma2_epsilon;

double b_sigma2_epsilon_update = 0.50*dot((y - off_set - x*beta_old - Z*theta_keep_old), (y - off_set - x*beta_old - Z*theta_keep_old)) + 
                                 b_sigma2_epsilon;

double sigma2_epsilon = 1.00/R::rgamma(a_sigma2_epsilon_update,
                                       (1.00/b_sigma2_epsilon_update));

return(sigma2_epsilon);

}





