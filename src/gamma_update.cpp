#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List gamma_update(arma::mat x,
                        arma::vec radius_seq,
                        arma::mat exposure,
                        arma::vec off_set,
                        arma::mat w,
                        int n_ind,
                        int n_grid,
                        int m,
                        int p_w,
                        arma::mat dists12,
                        arma::rowvec one_vec,
                        double sigma2_gamma,
                        arma::vec omega,
                        arma::vec lambda,
                        arma::vec beta, 
                        arma::vec theta,
                        arma::vec gamma_old,
                        arma::vec phi_star,
                        double rho_phi_old,
                        arma::mat phi_star_corr_inv,
                        Rcpp::List radius_Z_output,
                        arma::vec metrop_var_gamma,
                        arma::vec acctot_gamma){
  
double denom = 0.00;
double numer = 0.00;

arma::vec gamma = gamma_old;

for(int j = 0; j < p_w; ++j){
  
   //Second
   arma::mat Z = Rcpp::as<arma::mat>(radius_Z_output[1]);
   arma::mat G = Rcpp::as<arma::mat>(radius_Z_output[2]);
   arma::vec theta_keep = one_vec*(G*theta)/n_ind;
   
   denom = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
            -(0.50/sigma2_gamma)*pow(gamma(j), 2);
            
   Rcpp::List radius_Z_output_old = radius_Z_output;
    
   //First
   gamma(j) = R::rnorm(gamma_old(j),
                       sqrt(metrop_var_gamma(j)));
   radius_Z_output = create_radius_Z_fun(radius_seq,
                                         exposure,
                                         w,
                                         n_ind,
                                         m,
                                         dists12,
                                         gamma,
                                         phi_star,
                                         rho_phi_old,
                                         phi_star_corr_inv);
     
   Z = Rcpp::as<arma::mat>(radius_Z_output[1]);
   G = Rcpp::as<arma::mat>(radius_Z_output[2]);
   theta_keep = one_vec*(G*theta)/n_ind;
    
   numer = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
           -(0.50/sigma2_gamma)*pow(gamma(j), 2);
      
   /*Decision*/
   double ratio = exp(numer - denom);   
   double acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
        
     gamma(j) = gamma_old(j);
     radius_Z_output = radius_Z_output_old;
     acc = 0;
        
     }
   acctot_gamma(j) = acctot_gamma(j) + 
                     acc;
        
   }

return Rcpp::List::create(Rcpp::Named("gamma") = gamma,
                          Rcpp::Named("acctot_gamma") = acctot_gamma,
                          Rcpp::Named("radius_Z_output") = radius_Z_output);

}



