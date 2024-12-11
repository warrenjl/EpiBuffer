#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List neg_two_loglike_update(arma::vec y,
                                  arma::mat x,
                                  arma::vec off_set,
                                  arma::vec tri_als,
                                  int likelihood_indicator,
                                  int n_ind,
                                  int r,
                                  double sigma2_epsilon,
                                  arma::vec beta,
                                  arma::vec eta,
                                  arma::mat Z){

arma::vec log_density(n_ind); log_density.fill(0.00);

arma::vec mu = off_set +
               x*beta + 
               Z*eta;

arma::vec prob(n_ind); prob.fill(0.00);

if(likelihood_indicator == 0){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n_ind; ++j){
     log_density(j) = R::dbinom(y(j),
                                tri_als(j),
                                probs(j),
                                TRUE);
     }
  
  }

if(likelihood_indicator == 1){
  for(int j = 0; j < n_ind; ++j){
     log_density(j) = R::dnorm(y(j),
                               mu(j),
                               sqrt(sigma2_epsilon),
                               TRUE);
     }
  }

if(likelihood_indicator == 2){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n_ind; ++j){
     log_density(j) = R::dnbinom(y(j), 
                                 r, 
                                 (1.00 - probs(j)),        
                                 TRUE);
     }
  
  }

double neg_two_loglike = -2.00*sum(log_density);

return Rcpp::List::create(Rcpp::Named("neg_two_loglike") = neg_two_loglike,
                          Rcpp::Named("log_density") = log_density);

}

























































