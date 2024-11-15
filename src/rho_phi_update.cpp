#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho_phi_update(arma::vec radius_range,
                          int exposure_definition_indicator,
                          arma::mat exposure_dists,
                          int p_d,
                          int n_ind,
                          int n_grid,
                          int m,
                          int m_max,
                          int p_w,
                          arma::mat x,
                          arma::mat w,
                          arma::mat v,
                          arma::vec off_set,
                          arma::mat dists12,
                          arma::mat dists22,
                          double a_rho_phi,
                          double b_rho_phi,
                          arma::vec omega,
                          arma::vec lambda,
                          arma::vec beta, 
                          arma::vec eta,
                          arma::vec gamma,
                          arma::vec radius,
                          arma::vec theta,
                          double rho_phi_old,
                          arma::vec radius_trans,
                          arma::vec phi_star,
                          arma::vec phi_tilde,
                          Rcpp::List phi_star_corr_info,
                          arma::mat C,
                          arma::mat poly,
                          arma::vec exposure,
                          arma::mat Z,
                          double metrop_var_rho_phi,
                          int acctot_rho_phi){

/*Second*/
arma::vec radius_old = radius;
arma::vec theta_old = theta;
arma::vec radius_trans_old = radius_trans;
double rho_phi_trans_old = log(rho_phi_old);
arma::vec phi_tilde_old = phi_tilde;
Rcpp::List phi_star_corr_info_old = phi_star_corr_info;
arma::mat phi_star_corr_inv_old = phi_star_corr_info_old[0];
double phi_star_log_deter_corr_inv_old = phi_star_corr_info_old[1];
arma::mat C_old = C;
arma::mat poly_old = poly;
arma::vec exposure_old = exposure;
arma::mat Z_old = Z;

double denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) +
               0.50*phi_star_log_deter_corr_inv_old + 
               -0.50*dot(phi_star, (phi_star_corr_inv_old*phi_star)) +
               a_rho_phi*rho_phi_trans_old +
               -b_rho_phi*exp(rho_phi_trans_old);

/*First*/
double rho_phi_trans = R::rnorm(rho_phi_trans_old, 
                                sqrt(metrop_var_rho_phi));
double rho_phi = exp(rho_phi_trans);
phi_star_corr_info = spatial_corr_fun(rho_phi,
                                      dists22);
arma::mat phi_star_corr_inv = phi_star_corr_info[0];
double phi_star_log_deter_corr_inv = phi_star_corr_info[1];
C = exp(-rho_phi*dists12);
phi_tilde = C*(phi_star_corr_inv*phi_star);
radius_trans = (v*w)*gamma +
               v*phi_tilde;
Rcpp::NumericVector radius_trans_nv = Rcpp::NumericVector(radius_trans.begin(), 
                                                          radius_trans.end());
Rcpp::NumericVector radius_nv = Rcpp::pnorm(radius_trans_nv,
                                            0.00,
                                            1.00,
                                            true,
                                            false);
radius_nv = radius_nv*(radius_range(1) - radius_range(0)) + 
            radius_range(0); 

radius = arma::vec(Rcpp::as<std::vector<double>>(radius_nv));
arma::mat radius_mat(n_ind, m); radius_mat.fill(0.00);
for(int j = 0; j < n_ind; ++ j){
   radius_mat.row(j).fill(radius(j));
   }

for(int j = 0; j < (p_d + 1); ++j){
   poly.col(j) = pow((radius - radius_range(0))/(radius_range(1) - radius_range(0)), j);
   }
theta = poly*eta;

//Cumulative Counts
if(exposure_definition_indicator == 0){
  
  arma::umat comparison = ((v*exposure_dists) < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::sum(numeric_mat,
                       1);
  exposure = exposure/m_max;
  
  }

//Spherical
if(exposure_definition_indicator == 1){
  
  arma::mat corrs = 1.00 +
                    -1.50*((v*exposure_dists)/radius_mat) +
                    0.50*pow(((v*exposure_dists)/radius_mat), 3);
  arma::umat comparison = ((v*exposure_dists) < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  arma::mat prod = corrs%numeric_mat;
  exposure = arma::sum(prod,
                       1);
  exposure = exposure/m_max;
  
  }

//Presence/Absence
if(exposure_definition_indicator == 2){
  
  arma::umat comparison = ((v*exposure_dists) < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::max(numeric_mat,
                       1);
  
  }

for(int j = 0; j < (p_d + 1); ++j){
   Z.col(j) = exposure%poly.col(j);
   } 
double numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) +
               0.50*phi_star_log_deter_corr_inv + 
               -0.50*dot(phi_star, (phi_star_corr_inv*phi_star)) +
               a_rho_phi*rho_phi_trans +
               -b_rho_phi*exp(rho_phi_trans);

/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  radius = radius_old;
  theta = theta_old;
  rho_phi = rho_phi_old;
  radius_trans = radius_trans_old;
  phi_tilde = phi_tilde_old;
  phi_star_corr_info = phi_star_corr_info_old;
  C = C_old;
  poly = poly_old;
  exposure = exposure_old;
  Z = Z_old;
  acc = 0;
  
  }
acctot_rho_phi = acctot_rho_phi + 
                 acc;

return Rcpp::List::create(Rcpp::Named("rho_phi") = rho_phi,
                          Rcpp::Named("acctot_rho_phi") = acctot_rho_phi,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("phi_tilde") = phi_tilde,                  
                          Rcpp::Named("phi_star_corr_info") = phi_star_corr_info,
                          Rcpp::Named("C") = C,
                          Rcpp::Named("poly") = poly,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);

}



