#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List rho_theta_update(int m,
                            double l_rho_theta,
                            double u_rho_theta,
                            arma::vec theta,
                            double sigma2_theta,
                            double rho_theta_old,
                            Rcpp::List theta_corr_info,
                            double metrop_var_rho_theta,
                            int acctot_rho_theta){

/*Second*/
Rcpp::List theta_corr_info_old = theta_corr_info;
arma::mat theta_corr_inv_old = theta_corr_info_old[0];
double theta_log_deter_corr_inv_old = theta_corr_info_old[1];
double rho_theta_trans_old = log((rho_theta_old - l_rho_theta)/(u_rho_theta - rho_theta_old));

double denom = 0.50*m*log(1.00 - pow(rho_theta_old, 2.00)) +
               0.50*theta_log_deter_corr_inv_old + 
               -0.50*(1.00 - pow(rho_theta_old, 2.00))*dot(theta, (theta_corr_inv_old*theta))/sigma2_theta + 
               -rho_theta_trans_old +
               -2.00*log(1.00 + exp(-rho_theta_trans_old));

/*First*/
double rho_theta_trans = R::rnorm(rho_theta_trans_old, 
                                  sqrt(metrop_var_rho_theta));
double rho_theta = (u_rho_theta*exp(rho_theta_trans) + l_rho_theta)/(exp(rho_theta_trans) + 1.00);
theta_corr_info = temporal_corr_fun(m, 
                                    rho_theta);
arma::mat theta_corr_inv = theta_corr_info[0];
double theta_log_deter_corr_inv = theta_corr_info[1];

double numer = 0.50*m*log(1.00 - pow(rho_theta, 2.00)) +
               0.50*theta_log_deter_corr_inv + 
               -0.50*(1.00 - pow(rho_theta, 2.00))*dot(theta, (theta_corr_inv*theta))/sigma2_theta + 
               -rho_theta_trans +
               -2.00*log(1.00 + exp(-rho_theta_trans));

/*Decision*/
double ratio = exp(numer - denom);   
double acc = 1;
if(ratio < R::runif(0.00, 1.00)){
  
  rho_theta = rho_theta_old;
  theta_corr_info = theta_corr_info_old;
  acc = 0;
  
  }
acctot_rho_theta = acctot_rho_theta + 
                   acc;

return Rcpp::List::create(Rcpp::Named("rho_theta") = rho_theta,
                          Rcpp::Named("acctot_rho_theta") = acctot_rho_theta,
                          Rcpp::Named("theta_corr_info") = theta_corr_info);

}



