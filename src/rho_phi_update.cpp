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
                          int m,
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
                          arma::mat C,
                          arma::vec phi_tilde,
                          arma::vec delta_star_trans,
                          arma::vec delta_star,
                          arma::vec radius_pointer,
                          arma::mat G,
                          arma::vec radius,
                          arma::mat Z,
                          arma::vec theta_keep,
                          double metrop_var_rho_phi,
                          int acctot_rho_phi){

/*Second*/
double rho_phi_trans_old = log(rho_phi_old);
Rcpp::List phi_star_corr_info_old = phi_star_corr_info;
arma::mat phi_star_corr_inv_old = phi_star_corr_info_old[0];
double phi_star_log_deter_corr_inv_old = phi_star_corr_info_old[1];
arma::mat C_old = C;
arma::vec phi_tilde_old = phi_tilde;
arma::vec delta_star_trans_old = delta_star_trans;
arma::vec delta_star_old = delta_star;
arma::vec radius_pointer_old = radius_pointer;
arma::mat G_old = G;
arma::vec radius_old = radius;
arma::mat Z_old = Z;
arma::vec theta_keep_old = theta_keep;
  
double denom = -0.50*dot((lambda - off_set - x*beta - Z_old*theta_keep_old), (omega%(lambda - off_set - x*beta - Z_old*theta_keep_old))) +
               0.50*phi_star_log_deter_corr_inv_old + 
               -0.50*dot(phi_star, (phi_star_corr_inv_old*phi_star))/sigma2_phi +
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

//Start:  Previous Function
phi_tilde = C*(phi_star_corr_inv*phi_star);
delta_star_trans = w*gamma +
                   phi_tilde;
delta_star = 1.00/(1.00 + exp(-delta_star_trans));
radius_pointer = ceil(delta_star*m);
arma::uvec lt1 = find(radius_pointer < 1);
radius_pointer.elem(lt1).fill(1);
arma::uvec gtm = find(radius_pointer > m);
radius_pointer.elem(gtm).fill(m);
for(int j = 0; j < m; ++j){
  
   arma::uvec ej = find(radius_pointer == (j + 1));
   arma::colvec temp_col = G.col(j);
   temp_col.elem(ej).fill(1);
   G.col(j) = temp_col;
  
   }
arma::uvec radius_pointer_uvec = arma::conv_to<arma::uvec>::from(radius_pointer);
arma::mat temp_mat = exposure.cols(radius_pointer_uvec - 1);  
radius = radius_seq.elem(radius_pointer_uvec - 1);
Z.col(0) = temp_mat.diag(0);
theta_keep = one_vec*(G*theta)/n_ind;
//End:  Previous Function

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
  C = C_old;
  phi_tilde = phi_tilde_old;
  delta_star_trans = delta_star_trans_old;
  delta_star = delta_star_old;
  radius_pointer = radius_pointer_old;
  G = G_old;
  radius = radius_old;
  Z = Z_old;
  theta_keep = theta_keep_old;
  acc = 0;
  
  }
acctot_rho_phi = acctot_rho_phi + 
                 acc;

return Rcpp::List::create(Rcpp::Named("rho_phi") = rho_phi,
                          Rcpp::Named("acctot_rho_phi") = acctot_rho_phi,
                          Rcpp::Named("phi_star_corr_info") = phi_star_corr_info,
                          Rcpp::Named("C") = C,
                          Rcpp::Named("phi_tilde") = phi_tilde,
                          Rcpp::Named("delta_star_trans") = delta_star_trans,
                          Rcpp::Named("delta_star") = delta_star,
                          Rcpp::Named("radius_pointer") = radius_pointer,
                          Rcpp::Named("G") = G,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("Z") = Z,
                          Rcpp::Named("theta_keep") = theta_keep);

}



