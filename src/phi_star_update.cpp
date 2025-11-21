#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List phi_star_update(arma::vec radius_range,
                           int exposure_definition_indicator,
                           arma::mat v_exposure_dists,
                           int p_q,
                           int n_ind,
                           int n_grid,
                           int m,
                           double m_sd,
                           int p_w,
                           arma::mat x,
                           arma::mat v_q,
                           arma::mat v_w,
                           arma::vec v_index,
                           arma::vec off_set,
                           arma::vec omega,
                           arma::vec lambda,
                           arma::vec beta, 
                           arma::vec eta,
                           arma::vec gamma,
                           double tau_phi_old,
                           arma::vec radius,
                           arma::vec radius_trans,
                           arma::vec phi_star,
                           arma::vec phi_tilde,
                           arma::mat phi_star_corr_inv,
                           arma::mat C,
                           arma::vec exposure,
                           arma::mat Z,
                           arma::vec metrop_var_phi_star,
                           arma::vec acctot_phi_star){

double denom = 0.00;
double numer = 0.00;

for(int j = 0; j < n_grid; ++j){

   //Second
   arma::vec radius_old = radius;
   arma::vec radius_trans_old = radius_trans;
   arma::vec phi_star_old = phi_star;
   arma::vec phi_tilde_old = phi_tilde;
   arma::vec exposure_old = exposure;
   arma::mat Z_old = Z;
   
   denom = -0.50*dot((lambda - off_set - x*beta - Z_old*eta), (omega%(lambda - off_set - x*beta - Z_old*eta))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star))/pow(tau_phi_old, 2.00);
   
   //First
   phi_star(j) = R::rnorm(phi_star_old(j),
                          sqrt(metrop_var_phi_star(j)));
   
   phi_tilde = C*(phi_star_corr_inv*phi_star);
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
   for(int k = 0; k < n_ind; ++ k){
      radius_mat.row(k).fill(radius(k));
      }
   
   //Cumulative Counts
   if(exposure_definition_indicator == 0){
     
     arma::umat comparison = ((v_exposure_dists) < radius_mat);
     arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
     exposure = arma::sum(numeric_mat,
                          1);
     exposure = exposure/m_sd;
     
     }
   
   //Spherical
   if(exposure_definition_indicator == 1){
     
     arma::vec exposure_tmp(v_exposure_dists.n_rows, arma::fill::zeros);
     for(arma::uword i = 0; i < v_exposure_dists.n_rows; ++i){
       
        double sum_val = 0.0;
        for(arma::uword j = 0; j < v_exposure_dists.n_cols; ++j){
         
           double dist = v_exposure_dists(i,j);
           double rad = radius_mat(i,j);
         
           if(dist < rad){
           
             double fast = dist/rad;
             double fast3 = fast*fast*fast;
             sum_val += 1.0 - 1.5*fast + 0.5*fast3;
           
             }
         
        }
     exposure_tmp(i) = sum_val;
       
     }
   exposure = exposure_tmp/m_sd;
     
   }
   
   //Presence/Absence
   if(exposure_definition_indicator == 2){
     
     arma::umat comparison = ((v_exposure_dists) < radius_mat);
     arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
     exposure = arma::max(numeric_mat,
                          1);
     exposure = exposure/m_sd;
     
     }
   
   for(int k = 0; k < p_q; ++k){
      Z.col(k) = exposure%v_q.col(k);
      } 
   numer = -0.50*dot((lambda - off_set - x*beta - Z*eta), (omega%(lambda - off_set - x*beta - Z*eta))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star))/pow(tau_phi_old, 2.00);
           
   //Decision
   double ratio = exp(numer - denom);
   int acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
       
     radius = radius_old;
     radius_trans = radius_trans_old;
     phi_star(j) = phi_star_old(j);
     phi_tilde = phi_tilde_old;
     exposure = exposure_old;
     Z = Z_old;
     acc = 0;
     
     }
   acctot_phi_star(j) = acctot_phi_star(j) + 
                        acc;

   }
     
//Center on the fly
//phi_star = phi_star + 
//           -mean(phi_star);
//phi_tilde = C*(phi_star_corr_inv*phi_star);
//arma::vec phi_tilde_full(n_ind); phi_tilde_full.fill(0.00);
//for(int j = 0; j < n_ind; ++j){
//   phi_tilde_full(j) = phi_tilde(v_index(j));
//   }
//radius_trans = (v_w)*gamma +
//               phi_tilde_full;
//Rcpp::NumericVector radius_trans_nv = Rcpp::NumericVector(radius_trans.begin(), 
//                                                          radius_trans.end());
//Rcpp::NumericVector radius_nv = Rcpp::pnorm(radius_trans_nv,
//                                            0.00,
//                                            1.00,
//                                            true,
//                                            false);
//radius_nv = radius_nv*(radius_range(1) - radius_range(0)) + 
//            radius_range(0); 
//
//radius = arma::vec(Rcpp::as<std::vector<double>>(radius_nv));
//arma::mat radius_mat(n_ind, m); radius_mat.fill(0.00);
//for(int j = 0; j < n_ind; ++ j){
//   radius_mat.row(j).fill(radius(j));
//   }
//
//Cumulative Counts
//if(exposure_definition_indicator == 0){
//  
//  arma::umat comparison = ((v_exposure_dists) < radius_mat);
//  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
//  exposure = arma::sum(numeric_mat,
//                       1);
//  exposure = exposure/m_sd;
//  
//  }
//
//Spherical
//if(exposure_definition_indicator == 1){
//  
//  arma::vec exposure_tmp(v_exposure_dists.n_rows, arma::fill::zeros);
//  for(arma::uword i = 0; i < v_exposure_dists.n_rows; ++i){
//    
//     double sum_val = 0.0;
//     for(arma::uword j = 0; j < v_exposure_dists.n_cols; ++j){
//      
//        double dist = v_exposure_dists(i,j);
//        double rad = radius_mat(i,j);
//      
//        if(dist < rad){
//        
//          double fast = dist/rad;
//          double fast3 = fast*fast*fast;
//          sum_val += 1.0 - 1.5*fast + 0.5*fast3;
//        
//          }
//      
//        }
//     exposure_tmp(i) = sum_val;
//    
//     }
//  exposure = exposure_tmp/m_sd;
//  
//  }
//
//Presence/Absence
//if(exposure_definition_indicator == 2){
//  
//  arma::umat comparison = ((v_exposure_dists) < radius_mat);
//  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
//  exposure = arma::max(numeric_mat,
//                       1);
//  exposure = exposure/m_sd;
//  
//  }
//
//for(int j = 0; j < p_q; ++j){
//   Z.col(j) = exposure%v_q.col(j);
//   } 
//
return Rcpp::List::create(Rcpp::Named("phi_star") = phi_star,
                          Rcpp::Named("acctot_phi_star") = acctot_phi_star,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("radius_trans") = radius_trans,
                          Rcpp::Named("phi_tilde") = phi_tilde,
                          Rcpp::Named("exposure") = exposure,
                          Rcpp::Named("Z") = Z);
                          
}



