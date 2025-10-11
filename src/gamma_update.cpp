#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List gamma_update(arma::vec radius_range,
                        int exposure_definition_indicator,
                        arma::mat v_exposure_dists,
                        int p_q,
                        int n_ind,
                        int m,
                        int m_max,
                        int p_w,
                        arma::mat x,
                        arma::mat q,
                        arma::mat v_w,
                        arma::vec v_index,
                        arma::vec off_set,
                        arma::vec omega,
                        arma::vec lambda,
                        arma::vec beta, 
                        arma::vec eta,
                        arma::vec gamma_old,
                        arma::vec radius,
                        arma::vec theta,
                        arma::vec radius_trans,
                        arma::vec phi_tilde,
                        arma::vec exposure,
                        arma::mat Z,
                        arma::vec metrop_var_gamma,
                        arma::vec acctot_gamma){
  
double denom = 0.00;
double numer = 0.00;

arma::vec gamma = gamma_old;

for(int j = 0; j < p_w; ++j){
  
   //Second
   arma::vec radius_old = radius;
   arma::vec radius_trans_old = radius_trans;
   arma::vec exposure_old = exposure;
   arma::mat Z_old = Z;
   
   denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) +
            -0.50*pow(gamma(j), 2);
            
   //First
   gamma(j) = R::rnorm(gamma_old(j),
                       sqrt(metrop_var_gamma(j)));
   arma::vec phi_tilde_full(n_ind); phi_tilde_full.fill(0.00);
   for(int k = 0; k < n_ind; ++k){
      phi_tilde_full(k) = phi_tilde(v_index(k));
      }
   radius_trans = (v_w)*gamma +
                  phi_tilde_full;
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
   for(int k = 0; k < n_ind; ++k){
      radius_mat.row(k).fill(radius(k));
      }
     
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
       
     arma::mat fast = v_exposure_dists/radius_mat;
     arma::mat corrs = 1.00 +
                       -1.50*fast +
                       0.50*pow(fast, 3);
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
     
   for(int k = 0; k < p_q; ++k){
      Z.col(k) = exposure%q.col(k);
      }
   
   numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) +
           -0.50*pow(gamma(j), 2);
      
   /*Decision*/
   double ratio = exp(numer - denom);   
   double acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
        
     gamma(j) = gamma_old(j);
     radius = radius_old;
     radius_trans = radius_trans_old;
     exposure = exposure_old;
     Z = Z_old;
     acc = 0;
        
     }
   acctot_gamma(j) = acctot_gamma(j) + 
                     acc;
        
   }

return Rcpp::List::create(Rcpp::Named("gamma") = gamma,
                          Rcpp::Named("acctot_gamma") = acctot_gamma,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);

}

