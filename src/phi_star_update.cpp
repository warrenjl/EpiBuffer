#include "RcppArmadillo.h"
#include "EpiBuffer.h"
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
                           arma::rowvec one_vec,
                           arma::vec omega,
                           arma::vec lambda,
                           arma::vec beta, 
                           arma::vec theta,
                           arma::vec gamma,
                           arma::vec phi_star_old,
                           arma::mat phi_star_corr_inv,
                           arma::mat C,
                           arma::vec phi_tilde,
                           arma::vec delta_star_trans,
                           arma::vec delta_star,
                           arma::vec radius_pointer,
                           arma::mat G,
                           arma::vec radius,
                           arma::mat Z,
                           arma::vec theta_keep,
                           arma::vec metrop_var_phi_star,
                           arma::vec acctot_phi_star){

double denom = 0.00;
double numer = 0.00;

arma::vec phi_star = phi_star_old; 

for(int j = 0; j < n_grid; ++j){

   //Second
   arma::vec phi_tilde_old = phi_tilde;
   arma::vec delta_star_trans_old = delta_star_trans;
   arma::vec delta_star_old = delta_star;
   arma::vec radius_pointer_old = radius_pointer;
   arma::mat G_old = G;
   arma::vec radius_old = radius;
   arma::mat Z_old = Z;
   arma::vec theta_keep_old = theta_keep;
  
   denom = -0.50*dot((lambda - off_set - x*beta - Z_old*theta_keep_old), (omega%(lambda - off_set - x*beta - Z_old*theta_keep_old))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star));
   
   //First
   phi_star(j) = R::rnorm(phi_star_old(j),
                          sqrt(metrop_var_phi_star(j)));
   
   //Start:  Previous Function
   phi_tilde = C*(phi_star_corr_inv*phi_star);
   delta_star_trans = w*gamma +
                      phi_tilde;
   Rcpp::NumericVector delta_star_trans_nv = Rcpp::NumericVector(delta_star_trans.begin(), 
                                                                 delta_star_trans.end());
   Rcpp::NumericVector delta_star_nv = Rcpp::pnorm(delta_star_trans_nv,
                                                   0.00,
                                                   1.00,
                                                   true,
                                                   false);
   delta_star = arma::vec(Rcpp::as<std::vector<double>>(delta_star_nv));
   radius_pointer = ceil(delta_star*m);
   arma::uvec lt1 = find(radius_pointer < 1);
   radius_pointer.elem(lt1).fill(1);
   arma::uvec gtm = find(radius_pointer > m);
   radius_pointer.elem(gtm).fill(m);
   G.fill(0);
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
   
   numer = -0.50*dot((lambda - off_set - x*beta - Z*theta_keep), (omega%(lambda - off_set - x*beta - Z*theta_keep))) +
           -0.50*dot(phi_star, (phi_star_corr_inv*phi_star));
           
   //Decision
   double ratio = exp(numer - denom);
   int acc = 1;
   if(ratio < R::runif(0.00, 1.00)){
       
     phi_star(j) = phi_star_old(j);
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
   acctot_phi_star(j) = acctot_phi_star(j) + 
                        acc;

   }
      
return Rcpp::List::create(Rcpp::Named("phi_star") = phi_star,
                          Rcpp::Named("acctot_phi_star") = acctot_phi_star,
                          Rcpp::Named("phi_tilde") = phi_tilde,
                          Rcpp::Named("delta_star_trans") = delta_star_trans,
                          Rcpp::Named("delta_star") = delta_star,
                          Rcpp::Named("radius_pointer") = radius_pointer,
                          Rcpp::Named("G") = G,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("Z") = Z,
                          Rcpp::Named("theta_keep") = theta_keep);
                          
}



