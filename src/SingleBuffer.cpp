#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List SingleBuffer(int mcmc_samples,
                        arma::vec y,
                        arma::mat x,
                        arma::vec radius_range,
                        int exposure_definition_indicator,
                        arma::mat exposure_dists,
                        int p_d,
                        double metrop_var_radius,
                        int likelihood_indicator,
                        Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> trials = R_NilValue,
                        Rcpp::Nullable<double> a_r_prior = R_NilValue,
                        Rcpp::Nullable<double> b_r_prior = R_NilValue,
                        Rcpp::Nullable<double> a_sigma2_epsilon_prior = R_NilValue,
                        Rcpp::Nullable<double> b_sigma2_epsilon_prior = R_NilValue,
                        Rcpp::Nullable<double> sigma2_beta_prior = R_NilValue,
                        Rcpp::Nullable<double> sigma2_eta_prior = R_NilValue,
                        Rcpp::Nullable<double> r_init = R_NilValue,
                        Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
                        Rcpp::Nullable<Rcpp::NumericVector> eta_init = R_NilValue,
                        Rcpp::Nullable<double> radius_init = R_NilValue){

//Defining Parameters and Quantities of Interest
int n_ind = y.size();
int p_x = x.n_cols;
int m = exposure_dists.n_cols;

arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::mat beta(p_x, mcmc_samples); beta.fill(0.00);
arma::mat eta((p_d + 1), mcmc_samples); eta.fill(0.00);
arma::vec radius(mcmc_samples); radius.fill(0.00);
arma::mat theta_keep(1, mcmc_samples); theta_keep.fill(0.00);
arma::vec r(mcmc_samples); r.fill(0.00);
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

double sigma2_eta = 10000.00;
if(sigma2_eta_prior.isNotNull()){
  sigma2_eta = Rcpp::as<double>(sigma2_eta_prior);
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

eta.col(0).fill(0.00);
if(eta_init.isNotNull()){
  eta.col(0).fill(Rcpp::as<double>(eta_init));
  }

radius(0) = (radius_range(1) - radius_range(0))/2.00;
if(radius_init.isNotNull()){
  radius(0) = Rcpp::as<double>(radius_init);
  }
double radius_trans = log((radius(0) - radius_range(0))/(radius_range(1) - radius(0)));
arma::mat radius_mat(n_ind, m); radius_mat.fill(radius(0));
arma::vec exposure(n_ind); exposure.fill(0.00);
arma::vec poly(p_d + 1); poly.fill(0.00);
for(int j = 0; j < (p_d + 1); ++j){
   poly(j) = pow((radius(0) - radius_range(0))/(radius_range(1) - radius_range(0)), j);
   }
theta_keep.col(0) = dot(poly, eta.col(0));

//Cumulative Counts
if(exposure_definition_indicator == 0){

  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::sum(numeric_mat,
                       1);
  exposure = exposure/m;

  }

//Spherical
if(exposure_definition_indicator == 1){
  
  arma::mat corrs = 1.00 +
                    -1.50*(exposure_dists/radius_mat) +
                    0.50*pow((exposure_dists/radius_mat), 3);
  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  arma::mat prod = corrs%numeric_mat;
  exposure = arma::sum(prod,
                       1);
  exposure = exposure/m;
  
  }

//Presence/Absence
if(exposure_definition_indicator == 2){
  
  arma::umat comparison = (exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::max(numeric_mat,
                       1);
  
  }

arma::mat Z(n_ind, (p_d + 1));
for(int j = 0; j < (p_d + 1); ++j){
  Z.col(j) = exposure*poly(j);
  }

neg_two_loglike(0) = neg_two_loglike_update(y,
                                            x,
                                            off_set,
                                            tri_als,
                                            likelihood_indicator,
                                            n_ind,
                                            r(0),
                                            sigma2_epsilon(0),
                                            beta.col(0),
                                            eta.col(0),
                                            Z);

//Metropolis Settings
int acctot_radius = 0;

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
                                           eta.col(0),
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
                                               eta.col(j-1),
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
                                              eta.col(j-1),
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
                             eta.col(j-1),
                             Z);
   
   //eta Update
   eta.col(j) = eta_update(x,
                           off_set,
                           n_ind,
                           p_d,
                           sigma2_eta,
                           omega,
                           lambda,
                           beta.col(j),
                           Z);
   theta_keep.col(j) = dot(poly, eta.col(j));
   
   //radius Update
   Rcpp::List radius_output = radius_update(radius_range,
                                            exposure_definition_indicator,
                                            exposure_dists,
                                            p_d,
                                            n_ind,
                                            m,
                                            x,
                                            off_set,
                                            omega,
                                            lambda,
                                            beta.col(j),
                                            eta.col(j),
                                            radius_trans,
                                            radius(j-1),
                                            poly,
                                            exposure,
                                            Z,
                                            theta_keep.col(j),
                                            metrop_var_radius,
                                            acctot_radius);
   
   radius(j) = Rcpp::as<double>(radius_output[0]);
   acctot_radius = Rcpp::as<int>(radius_output[1]);
   radius_trans = Rcpp::as<double>(radius_output[2]);
   poly = Rcpp::as<arma::vec>(radius_output[3]);
   exposure = Rcpp::as<arma::vec>(radius_output[4]);
   Z = Rcpp::as<arma::mat>(radius_output[5]);
   theta_keep.col(j) = Rcpp::as<arma::vec>(radius_output[6]);
   
   if(likelihood_indicator == 2){
     
     //r Update
     r(j) = r_update(y,
                     x,
                     off_set,
                     n_ind,
                     a_r,
                     b_r,
                     beta.col(j),
                     eta.col(j),
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
                                              eta.col(j),
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
                                               eta.col(j),
                                               Z);
  
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     
     double accrate_radius = round(100*(acctot_radius/(double)j));
     Rcpp::Rcout << "radius Acceptance: " << accrate_radius << "%" << std::endl;
     
     Rcpp::Rcout << "***********************" << std::endl;
    
     }
   
   }
                                  
return Rcpp::List::create(Rcpp::Named("r") = r,
                          Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                          Rcpp::Named("beta") = beta,
                          Rcpp::Named("eta") = eta,
                          Rcpp::Named("radius") = radius,
                          Rcpp::Named("theta_keep") = theta_keep,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike);

}