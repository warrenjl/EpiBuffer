#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List phi_star_update(arma::mat x,
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
                           arma::vec omega,
                           arma::vec lambda,
                           arma::vec beta, 
                           arma::vec theta,
                           arma::vec gamma,
                           arma::vec phi_star_old,
                           double sigma2_phi_old,
                           double rho_phi_old,
                           arma::mat phi_star_corr_inv,
                           Rcpp::List radius_Z_output,
                           arma::vec metrop_var_phi_star,
                           arma::vec acctot_phi_star){

double denom = 0.00;
double numer = 0.00;

arma::vec phi_star = phi_star_old; 

for(int j = 0; j < n_grid; ++j){

   //Second
   arma::mat Z = Rcpp::as<arma::mat>(radius_Z_output[1]);
   arma::mat G = Rcpp::as<arma::mat>(radius_Z_output[2]);
   arma::vec theta_keep = one_vec*(G*theta)/n_ind;
  
   denom = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
           -(0.50/sigma2_phi_old)*dot(phi_star, (phi_star_corr_inv*phi_star));
   
   Rcpp::List radius_Z_output_old = radius_Z_output;
      
   //First
   phi_star(j) = R::rnorm(phi_star_old(j),
                          sqrt(metrop_var_phi_star(j)));
   Rcpp::List radius_Z_output = create_radius_Z_fun(radius_seq,
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
           -(0.50/sigma2_phi_old)*dot(phi_star, (phi_star_corr_inv*phi_star));
           
   //Decision
   double ratio = exp(numer - denom);
   int acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
       
     phi_star(j) = phi_star_old(j);
     radius_Z_output = radius_Z_output_old;
     acc = 0;
     
     }
   acctot_phi_star(j) = acctot_phi_star(j) + 
                        acc;

   }
      
return Rcpp::List::create(Rcpp::Named("phi_star") = phi_star,
                          Rcpp::Named("acctot_phi_star") = acctot_phi_star,
                          Rcpp::Named("radius_Z_output") = radius_Z_output);
                          
}



