#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List phi_star_update(arma::vec radius_range,
                           int exposure_definition_indicator,
                           arma::mat v_exposure_dists,
                           int p_d,
                           int n_ind,
                           int n_grid,
                           int m,
                           int m_max,
                           int p_w,
                           arma::mat x,
                           arma::mat v_w,
                           arma::mat v,
                           arma::vec off_set,
                           arma::vec omega,
                           arma::vec lambda,
                           arma::vec beta, 
                           arma::vec eta,
                           arma::vec gamma,
                           arma::vec radius,
                           arma::vec theta,
                           arma::vec radius_trans,
                           arma::vec phi_star,
                           arma::vec phi_tilde,
                           arma::mat phi_star_corr_inv,
                           arma::mat C,
                           arma::mat poly,
                           arma::vec exposure,
                           arma::mat Z,
                           arma::vec metrop_var_phi_star,
                           arma::vec acctot_phi_star){

double denom = 0.00;
double numer = 0.00;

for(int j = 0; j < n_grid; ++j){

   //Second
   arma::vec radius_old = radius;
   arma::vec theta_old = theta;
   arma::vec radius_trans_old = radius_trans;
   arma::vec phi_star_old = phi_star;
   arma::vec phi_tilde_old = phi_tilde;
   arma::mat poly_old = poly;
   arma::vec exposure_old = exposure;
   arma::mat Z_old = Z;
   
   denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star));
   
   //First
   phi_star(j) = R::rnorm(phi_star_old(j),
                          sqrt(metrop_var_phi_star(j)));
   
   phi_tilde = C*(phi_star_corr_inv*phi_star);
   radius_trans = (v_w)*gamma +
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
   for(int k = 0; k < n_ind; ++ k){
      radius_mat.row(k).fill(radius(k));
      }
   
   for(int k = 0; k < (p_d + 1); ++k){
      poly.col(k) = pow((radius - radius_range(0))/(radius_range(1) - radius_range(0)), k);
      }
   theta = poly*eta;
   
   //Cumulative Counts
   if(exposure_definition_indicator == 0){
     
     arma::umat comparison = ((v_exposure_dists) < radius_mat);
     arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
     exposure = arma::sum(numeric_mat,
                          1);
     exposure = exposure/m_max;
     
     }
   
   //Spherical
   if(exposure_definition_indicator == 1){
     
     arma::mat corrs = 1.00 +
                       -1.50*((v_exposure_dists)/radius_mat) +
                       0.50*pow(((v_exposure_dists)/radius_mat), 3);
     arma::umat comparison = ((v_exposure_dists) < radius_mat);
     arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
     arma::mat prod = corrs%numeric_mat;
     exposure = arma::sum(prod,
                          1);
     exposure = exposure/m_max;
     
     }
   
   //Presence/Absence
   if(exposure_definition_indicator == 2){
     
     arma::umat comparison = ((v_exposure_dists) < radius_mat);
     arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
     exposure = arma::max(numeric_mat,
                          1);
     
     }
   
   for(int k = 0; k < (p_d + 1); ++k){
      Z.col(k) = exposure%poly.col(k);
      } 
   numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star));
           
   //Decision
   double ratio = exp(numer - denom);
   int acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
       
     radius = radius_old;
     theta = theta_old;
     radius_trans = radius_trans_old;
     phi_star(j) = phi_star_old(j);
     phi_tilde = phi_tilde_old;
     poly = poly_old;
     exposure = exposure_old;
     Z = Z_old;
     acc = 0;
     
     }
   acctot_phi_star(j) = acctot_phi_star(j) + 
                        acc;

   }
      
return Rcpp::List::create(Rcpp::Named("phi_star") = phi_star,
                          Rcpp::Named("acctot_phi_star") = acctot_phi_star,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("phi_tilde") = phi_tilde,
                          Rcpp::Named("poly") = poly,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);
                          
}



