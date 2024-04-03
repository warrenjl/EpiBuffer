#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho_phi_update(arma::mat x,
                          arma::vec radius_seq,
                          arma::mat exposure,
                          arma::vec off_set,
                          arma::mat w,
                          int n_ind,
                          int n_grid,
                          int m,
                          int p_w,
                          arma::mat dists12,
                          arma::mat dists22,
                          arma::rowvec one_vec,
                          double a_rho_phi,
                          double b_rho_phi,
                          arma::vec omega,
                          arma::vec lambda,
                          arma::vec beta, 
                          arma::vec theta,
                          arma::vec gamma,
                          arma::vec phi_star,
                          double sigma2_phi,
                          double rho_phi_old,
                          Rcpp::List phi_star_corr_info,
                          Rcpp::List radius_Z_output,
                          double metrop_var_rho_phi,
                          int acctot_rho_phi){

/*Second*/
double rho_phi_trans_old = log(rho_phi_old);
Rcpp::List phi_star_corr_info_old = phi_star_corr_info;
arma::mat phi_star_corr_inv_old = phi_star_corr_info_old[0];
double phi_star_log_deter_corr_inv_old = phi_star_corr_info_old[1];
  
arma::mat Z = Rcpp::as<arma::mat>(radius_Z_output[1]);
arma::mat G = Rcpp::as<arma::mat>(radius_Z_output[2]);
arma::vec theta_keep = one_vec*(G*theta)/n_ind;

double denom = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
               0.50*phi_star_log_deter_corr_inv_old + 
               -0.50*dot(phi_star, (phi_star_corr_inv_old*phi_star))/sigma2_phi +
               a_rho_phi*rho_phi_trans_old +
               -b_rho_phi*exp(rho_phi_trans_old);

Rcpp::List radius_Z_output_old = radius_Z_output;

/*First*/
double rho_phi_trans = R::rnorm(rho_phi_trans_old, 
                                sqrt(metrop_var_rho_phi));
double rho_phi = exp(rho_phi_trans);
phi_star_corr_info = spatial_corr_fun(rho_phi,
                                      dists22);
arma::mat phi_star_corr_inv = phi_star_corr_info[0];
double phi_star_log_deter_corr_inv = phi_star_corr_info[1];

radius_Z_output = create_radius_Z_fun(radius_seq,
                                      exposure,
                                      w,
                                      n_ind,
                                      m,
                                      dists12,
                                      gamma,
                                      phi_star,
                                      rho_phi,
                                      phi_star_corr_inv);
Z = Rcpp::as<arma::mat>(radius_Z_output[1]);
G = Rcpp::as<arma::mat>(radius_Z_output[2]);
theta_keep = one_vec*(G*theta)/n_ind;

double numer = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
               0.50*phi_star_log_deter_corr_inv + 
               -0.50*dot(phi_star, (phi_star_corr_inv*phi_star))/sigma2_phi +
               a_rho_phi*rho_phi_trans +
               -b_rho_phi*exp(rho_phi_trans);

/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  rho_phi = rho_phi_old;
  phi_star_corr_info = phi_star_corr_info_old;
  radius_Z_output = radius_Z_output_old;
  acc = 0;
  
  }
acctot_rho_phi = acctot_rho_phi + 
                 acc;

return Rcpp::List::create(Rcpp::Named("rho_phi") = rho_phi,
                          Rcpp::Named("acctot_rho_phi") = acctot_rho_phi,
                          Rcpp::Named("phi_star_corr_info") = phi_star_corr_info,
                          Rcpp::Named("radius_Z_output") = radius_Z_output);

}



