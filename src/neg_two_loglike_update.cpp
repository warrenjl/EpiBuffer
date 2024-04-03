#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              arma::vec off_set,
                              arma::vec tri_als,
                              int likelihood_indicator,
                              int n_ind,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,
                              arma::vec theta_keep,
                              arma::mat Z){

arma::vec dens(n_ind); dens.fill(0.00);

arma::vec mu = off_set +
               x*beta + 
               Z*theta_keep;

arma::vec prob(n_ind); prob.fill(0.00);

if(likelihood_indicator == 0){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n_ind; ++j){
     dens(j) = R::dbinom(y(j),
                         tri_als(j),
                         probs(j),
                         TRUE);
     }
  
  }

if(likelihood_indicator == 1){
  for(int j = 0; j < n_ind; ++j){
     dens(j) = R::dnorm(y(j),
                        mu(j),
                        sqrt(sigma2_epsilon),
                        TRUE);
     }
  }

if(likelihood_indicator == 2){
  
  arma::vec probs = exp(mu)/(1.00 + exp(mu));
  for(int j = 0; j < n_ind; ++j){
     dens(j) = R::dnbinom(y(j), 
                          r, 
                          (1.00 - probs(j)),        
                          TRUE);
     }
  
  }

double neg_two_loglike = -2.00*sum(dens);

return neg_two_loglike;

}

























































