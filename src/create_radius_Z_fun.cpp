#include "RcppArmadillo.h"
#include "SpBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List create_radius_Z_fun(arma::vec radius_seq,
                               arma::mat exposure,
                               arma::mat w,
                               int n_ind,
                               int m,
                               arma::mat dists12,
                               arma::vec gamma,
                               arma::vec phi_star, 
                               double rho_phi,
                               arma::mat phi_star_corr_inv){
  
arma::mat C = exp(-rho_phi*dists12);
arma::vec phi_tilde = C*(phi_star_corr_inv*phi_star);
arma::vec delta_star_trans = w*gamma +
                             phi_tilde;
arma::vec delta_star = 1.00/(1.00 + exp(-delta_star_trans));
arma::vec radius_pointer = ceil(delta_star*m);
arma::uvec lt1 = find(radius_pointer < 1);
radius_pointer.elem(lt1).fill(1);
arma::uvec gtm = find(radius_pointer > m);
radius_pointer.elem(gtm).fill(m);
  
arma::mat G(n_ind, m); G.fill(0);
for(int j = 0; j < m; ++j){
    
   arma::uvec ej = find(radius_pointer == (j + 1));
   arma::colvec temp_col = G.col(j);
   temp_col.elem(ej).fill(1);
   G.col(j) = temp_col;
    
   }
  
arma::vec radius(n_ind); radius.fill(0.00);
arma::mat Z(n_ind, 1); Z.fill(0.00);
for(int j = 0; j < n_ind; ++j){
    
   radius(j) = radius_seq(radius_pointer(j) - 1);
   Z.row(j) = exposure(j, (radius_pointer(j) - 1));   
    
   }
  
return Rcpp::List::create(Rcpp::Named("radius") = radius,
                          Rcpp::Named("Z") = Z,
                          Rcpp::Named("G") = G);
  
}
