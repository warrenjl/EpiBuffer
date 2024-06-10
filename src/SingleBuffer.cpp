#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List SingleBuffer(int mcmc_samples,
                        arma::vec y,
                        arma::mat x,
                        arma::vec radius_seq,
                        arma::mat exposure,
                        double metrop_var_rho_theta,
                        int likelihood_indicator,
                        Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> trials = R_NilValue,
                        Rcpp::Nullable<double> a_r_prior = R_NilValue,
                        Rcpp::Nullable<double> b_r_prior = R_NilValue,
                        Rcpp::Nullable<double> a_sigma2_epsilon_prior = R_NilValue,
                        Rcpp::Nullable<double> b_sigma2_epsilon_prior = R_NilValue,
                        Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                        Rcpp::Nullable<double> a_sigma2_theta_prior = R_NilValue,
                        Rcpp::Nullable<double> b_sigma2_theta_prior = R_NilValue,
                        Rcpp::Nullable<double> l_rho_theta_prior = R_NilValue,
                        Rcpp::Nullable<double> u_rho_theta_prior = R_NilValue,
                        Rcpp::Nullable<double> r_init = R_NilValue,
                        Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
                        Rcpp::Nullable<double> theta_keep_init = R_NilValue,
                        Rcpp::Nullable<double> sigma2_theta_init = R_NilValue,
                        Rcpp::Nullable<double> rho_theta_init = R_NilValue,
                        Rcpp::Nullable<double> radius_pointer_init = R_NilValue){

//Defining Parameters and Quantities of Interest
int p_x = x.n_cols;
int m = exposure.n_cols;
int n_ind = exposure.n_rows;

arma::vec r(mcmc_samples); r.fill(0.00);
arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::mat beta(p_x, mcmc_samples); beta.fill(0.00);
arma::mat theta(m, mcmc_samples); theta.fill(0.00);
arma::vec sigma2_theta(mcmc_samples); sigma2_theta.fill(0.00);
arma::vec rho_theta(mcmc_samples); rho_theta.fill(0.00);
arma::mat radius(1, mcmc_samples); radius.fill(0.00);
arma::mat theta_keep(1, mcmc_samples); theta_keep.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);

arma::vec off_set(n_ind); off_set.fill(0.00);
if(offset.isNotNull()){
  off_set = Rcpp::as<arma::vec>(offset);
  }

arma::vec tri_als(n_ind); tri_als.fill(1);
if(trials.isNotNull()){
  tri_als = Rcpp::as<arma::vec>(trials);
  }

//Prior Information
int a_r = 1;
if(a_r_prior.isNotNull()){
  a_r = Rcpp::as<int>(a_r_prior);
  }

int b_r = 100;
if(b_r_prior.isNotNull()){
  b_r = Rcpp::as<int>(b_r_prior);
  }

double a_sigma2_epsilon = 0.01;
if(a_sigma2_epsilon_prior.isNotNull()){
  a_sigma2_epsilon = Rcpp::as<double>(a_sigma2_epsilon_prior);
  }

double b_sigma2_epsilon = 0.01;
if(b_sigma2_epsilon_prior.isNotNull()){
  b_sigma2_epsilon = Rcpp::as<double>(b_sigma2_epsilon_prior);
  }

double sigma2_beta = 10000.00;
if(sigma2_beta_prior.isNotNull()){
  sigma2_beta = Rcpp::as<double>(sigma2_beta_prior);
  }

double a_sigma2_theta = 3.00;
if(a_sigma2_theta_prior.isNotNull()){
  a_sigma2_theta = Rcpp::as<double>(a_sigma2_theta_prior);
  }

double b_sigma2_theta = 2.00;
if(b_sigma2_theta_prior.isNotNull()){
  b_sigma2_theta = Rcpp::as<double>(b_sigma2_theta_prior);
  }

double l_rho_theta = 0.00;
if(l_rho_theta_prior.isNotNull()){
  l_rho_theta = Rcpp::as<double>(l_rho_theta_prior);
  }

double u_rho_theta = 1.00;
if(u_rho_theta_prior.isNotNull()){
  u_rho_theta = Rcpp::as<double>(u_rho_theta_prior);
  }

//Initial Values
r(0) = b_r;
if(r_init.isNotNull()){
  r(0) = Rcpp::as<int>(r_init);
  }

sigma2_epsilon(0) = var(y);
if(sigma2_epsilon_init.isNotNull()){
  sigma2_epsilon(0) = Rcpp::as<double>(sigma2_epsilon_init);
  }

beta.col(0).fill(0.00);
if(beta_init.isNotNull()){
  beta.col(0) = Rcpp::as<arma::vec>(beta_init);
  }

theta.col(0).fill(0.00);
if(theta_keep_init.isNotNull()){
  theta.col(0).fill(Rcpp::as<double>(theta_keep_init));
  }

sigma2_theta(0) = 0.01;
if(sigma2_theta_init.isNotNull()){
  sigma2_theta(0) = Rcpp::as<double>(sigma2_theta_init);
  }

rho_theta(0) = 0.50;
if(rho_theta_init.isNotNull()){
  rho_theta(0) = Rcpp::as<double>(rho_theta_init);
  }

Rcpp::List theta_corr_info = temporal_corr_fun(m,
                                               rho_theta(0));

arma::vec radius_pointer(1); radius_pointer(0) = ceil(m/2);
if(radius_pointer_init.isNotNull()){
  radius_pointer(0) = Rcpp::as<double>(radius_pointer_init);
  }
arma::uvec radius_pointer_uvec = arma::conv_to<arma::uvec>::from(radius_pointer);
radius.col(0) = radius_seq.elem(radius_pointer_uvec - 1);
arma::mat Z(n_ind, 1); Z.fill(0.00);
Z.col(0) = exposure.cols(radius_pointer_uvec - 1);
theta_keep.col(0) = theta(0, (radius_pointer(0) - 1));

neg_two_loglike(0) = neg_two_loglike_update(y,
                                            x,
                                            off_set,
                                            tri_als,
                                            likelihood_indicator,
                                            n_ind,
                                            r(0),
                                            sigma2_epsilon(0),
                                            beta.col(0),
                                            theta_keep.col(0),
                                            Z);

//Metropolis Settings
int acctot_rho_theta = 0;

//Main Sampling Loop
arma::vec omega(n_ind); omega.fill(0.00);
arma::vec lambda = y;
if(likelihood_indicator == 2){
  
  Rcpp::List latent_output = latent_update(y,
                                           x,
                                           off_set,
                                           tri_als,
                                           likelihood_indicator,
                                           n_ind,
                                           r(0),
                                           beta.col(0),
                                           theta_keep.col(0),
                                           Z);
  omega = Rcpp::as<arma::vec>(latent_output[0]);
  lambda = Rcpp::as<arma::vec>(latent_output[1]);
  
  }

for(int j = 1; j < mcmc_samples; ++j){
   
   if(likelihood_indicator == 1){
  
     //sigma2_epsilon Update
     sigma2_epsilon(j) = sigma2_epsilon_update(y,
                                               x,
                                               off_set,
                                               n_ind,
                                               a_sigma2_epsilon,
                                               b_sigma2_epsilon,
                                               beta.col(j-1),
                                               theta_keep.col(j-1),
                                               Z);
     omega.fill(1.00/sigma2_epsilon(j));
     
     }
    
   if(likelihood_indicator == 0){
    
     //latent parameters Update
     Rcpp::List latent_output = latent_update(y,
                                              x,
                                              off_set,
                                              tri_als,
                                              likelihood_indicator,
                                              n_ind,
                                              r(j-1),
                                              beta.col(j-1),
                                              theta_keep.col(j-1),
                                              Z);
     omega = Rcpp::as<arma::vec>(latent_output[0]);
     lambda = Rcpp::as<arma::vec>(latent_output[1]);
     
     }
  
   //beta Update
   beta.col(j) = beta_update(x,
                             off_set,
                             n_ind,
                             p_x,
                             sigma2_beta,
                             omega,
                             lambda,
                             theta_keep.col(j-1),
                             Z);
   
   //theta, theta_keep Update
   Rcpp::List theta_output = theta_single_update(x, 
                                                 off_set,
                                                 n_ind,
                                                 m,
                                                 omega,
                                                 lambda,
                                                 beta.col(j),
                                                 sigma2_theta(j-1),
                                                 rho_theta(j-1),
                                                 theta_corr_info[0],
                                                 radius_pointer,
                                                 Z);
   theta.col(j) = Rcpp::as<arma::vec>(theta_output[0]);
   theta_keep.col(j) = Rcpp::as<arma::vec>(theta_output[1]);
   
   //sigma2_theta Update
   sigma2_theta(j) = sigma2_theta_update(m,
                                         a_sigma2_theta,
                                         b_sigma2_theta,
                                         theta.col(j),
                                         rho_theta(j-1),
                                         theta_corr_info[0]);
   
   //rho_theta Update
   Rcpp::List rho_theta_output = rho_theta_update(m,
                                                  l_rho_theta,
                                                  u_rho_theta,
                                                  theta.col(j),
                                                  sigma2_theta(j),
                                                  rho_theta(j-1),
                                                  theta_corr_info,
                                                  metrop_var_rho_theta,
                                                  acctot_rho_theta);
   rho_theta(j) = Rcpp::as<double>(rho_theta_output[0]);
   acctot_rho_theta = Rcpp::as<int>(rho_theta_output[1]);
   theta_corr_info = Rcpp::as<Rcpp::List>(rho_theta_output[2]);
   
   //radius Update
   
   
   if(likelihood_indicator == 2){
     
     //r Update
     r(j) = r_update(y,
                     x,
                     off_set,
                     n_ind,
                     a_r,
                     b_r,
                     beta.col(j),
                     theta_keep.col(j),
                     Z);
     
     //latent parameters Update
     Rcpp::List latent_output = latent_update(y,
                                              x,
                                              off_set,
                                              tri_als,
                                              likelihood_indicator,
                                              n_ind,
                                              r(j),
                                              beta.col(j),
                                              theta_keep.col(j),
                                              Z);
     omega = Rcpp::as<arma::vec>(latent_output[0]);
     lambda = Rcpp::as<arma::vec>(latent_output[1]);
     
     }
   
   //neg_two_loglike Update
   neg_two_loglike(j) = neg_two_loglike_update(y,
                                               x,
                                               off_set,
                                               tri_als,
                                               likelihood_indicator,
                                               n_ind,
                                               r(j),
                                               sigma2_epsilon(j),
                                               beta.col(j),
                                               theta_keep.col(j),
                                               Z);
  
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     
     double accrate_rho_theta = round(100*(acctot_rho_theta/(double)j));
     Rcpp::Rcout << "rho_theta Acceptance: " << accrate_rho_theta << "%" << std::endl;
     
     Rcpp::Rcout << "******************************" << std::endl;
    
     }
   
   }
                                  
return Rcpp::List::create(Rcpp::Named("r") = r,
                          Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                          Rcpp::Named("beta") = beta,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("sigma2_theta") = sigma2_theta,
                          Rcpp::Named("rho_theta") = rho_theta,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("theta_keep") = theta_keep,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike);

}