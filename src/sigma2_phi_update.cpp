#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List sigma2_phi_update(int n_grid,
                             double b_sigma2_phi,
                             arma::vec phi_star,
                             double sigma2_phi_old,
                             arma::mat phi_star_corr_inv,
                             double metrop_delta_sigma_phi,
                             int acctot_sigma_phi){

/*Second*/
double sigma_phi_old = sqrt(sigma2_phi_old);
double denom = -(0.50*n_grid)*log(sigma2_phi_old) +
               -(0.50/sigma2_phi_old)*dot(phi_star, (phi_star_corr_inv*phi_star));
      
/*First*/         
double sigma_phi = R::runif((sigma_phi_old - metrop_delta_sigma_phi*sqrt(b_sigma2_phi)), 
                            (sigma_phi_old + metrop_delta_sigma_phi*sqrt(b_sigma2_phi)));
if(sigma_phi < 0){
  sigma_phi = abs(sigma_phi);
  }
if(sigma_phi > sqrt(b_sigma2_phi)){
  sigma_phi = 2*sqrt(b_sigma2_phi) - 
              sigma_phi;
  }
double sigma2_phi = pow(sigma_phi, 2);
double numer = -(0.50*n_grid)*log(sigma2_phi) +
               -(0.50/sigma2_phi)*dot(phi_star, (phi_star_corr_inv*phi_star));
  
/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
    
  sigma2_phi = sigma2_phi_old;
  acc = 0;
    
  }
acctot_sigma_phi = acctot_sigma_phi + 
                   acc;
  
return Rcpp::List::create(Rcpp::Named("sigma2_phi") = sigma2_phi,
                          Rcpp::Named("acctot_sigma_phi") = acctot_sigma_phi);

}




