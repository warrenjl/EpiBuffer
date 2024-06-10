#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List latent_update(arma::vec y,
                         arma::mat x,
                         arma::vec off_set,
                         arma::vec tri_als,
                         int likelihood_indicator,
                         int n_ind,
                         int r_old,
                         arma::vec beta_old,
                         arma::vec theta_keep_old,
                         arma::mat Z){

arma::vec mean_omega = off_set +
                       x*beta_old + 
                       Z*theta_keep_old;

arma::vec input0 = tri_als;
arma::vec input2 = (r_old + y);
  
arma::vec omega(n_ind); omega.fill(0.00);
arma::vec lambda(n_ind); lambda.fill(0.00);
  
if(likelihood_indicator == 0){
    
  omega = rcpp_pgdraw(input0,
                      mean_omega);
  lambda = (y - 0.50)/omega;
    
  } 
  
if(likelihood_indicator == 2){
    
  omega = rcpp_pgdraw(input2,
                      mean_omega);
  lambda = 0.50*(y - r_old)/omega;
    
  }
  
return Rcpp::List::create(Rcpp::Named("omega") = omega,
                          Rcpp::Named("lambda") = lambda);

}
































































