#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List radius_update(arma::vec y,
                         arma::mat x,
                         arma::vec radius_seq,
                         arma::mat exposure,
                         arma::vec off_set,
                         arma::vec tri_als,
                         int likelihood_indicator,
                         int n_ind,
                         int m,
                         int r_old,
                         double sigma2_epsilon,
                         arma::vec beta, 
                         arma::vec theta,
                         arma::mat Z,
                         arma::vec theta_keep){

arma::vec radius_log_vec(n_ind); radius_log_vec.fill(0.00);
arma::vec radius_log_val(m); radius_log_val.fill(0.00);

for(int j = 0; j < m; ++j){

   arma::vec mu = off_set +
                  x*beta + 
                  exposure.col(j)*theta(j);
  
   if(likelihood_indicator == 0){
    
     arma::vec probs = exp(mu)/(1.00 + exp(mu));
     for(int k = 0; k < n_ind; ++k){
        radius_log_vec(k) = R::dbinom(y(k),
                                      tri_als(k),
                                      probs(k),
                                      TRUE);
        }
     
     }
   
   if(likelihood_indicator == 1){
     for(int k = 0; k < n_ind; ++k){
        radius_log_vec(k) = R::dnorm(y(k),
                                     mu(k),
                                     sqrt(sigma2_epsilon),
                                     TRUE);
        }
     }
   
   if(likelihood_indicator == 2){
     
     arma::vec probs = exp(mu)/(1.00 + exp(mu));
     for(int k = 0; k < n_ind; ++k){
        radius_log_vec(k) = R::dnbinom(y(k), 
                                       r_old, 
                                       (1.00 - probs(k)),        
                                       TRUE);
        }
     
     }
   
   radius_log_val(j) = sum(radius_log_vec);
    
   }

arma::vec radius_probs(m); radius_probs.fill(0.00);
for(int j = 0; j < m; ++j){

   radius_probs(j) = 1.00/sum(exp(radius_log_val - radius_log_val(j)));
   if(arma::is_finite(radius_probs(j)) == 0){
     radius_probs(j) = 0.00;  /*Computational Correction*/
     }
   
   }
  
IntegerVector sample_set = seq(1, m);
arma::vec radius_pointer(1);
radius_pointer(0) = sampleRcpp(wrap(sample_set), 
                               1, 
                               TRUE, 
                               wrap(radius_probs))(0);
arma::uvec radius_pointer_uvec = arma::conv_to<arma::uvec>::from(radius_pointer);
arma::vec radius = radius_seq.elem(radius_pointer_uvec - 1);
Z.fill(0.00);
Z.cols(radius_pointer_uvec - 1) = exposure.cols(radius_pointer_uvec - 1);
theta_keep = theta.elem(radius_pointer_uvec - 1);

return Rcpp::List::create(Rcpp::Named("radius_pointer") = radius_pointer,
                          Rcpp::Named("radius_pointer_uvec") = radius_pointer_uvec,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("Z") = Z,
                          Rcpp::Named("theta_keep") = theta_keep);
                          
}



