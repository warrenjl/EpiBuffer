#include "RcppArmadillo.h"
#include "EpiBuffer.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List SingleBuffer(int mcmc_samples,
                        arma::vec y,
                        arma::mat x,
                        arma::mat q,
                        Rcpp::IntegerVector v,
                        arma::vec radius_range,
                        int exposure_definition_indicator,
                        arma::mat exposure_dists,
                        double metrop_var_radius,
                        int likelihood_indicator,
                        Rcpp::Nullable<int> waic_info_indicator = R_NilValue,
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
int p_q = q.n_cols;
int m = exposure_dists.n_cols;
int waic_info_ind = 0;  //No by Default
if(waic_info_indicator.isNotNull()){
  waic_info_ind = Rcpp::as<int>(waic_info_indicator);
  }

arma::vec r(mcmc_samples); r.fill(0.00);
arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::mat beta(p_x, mcmc_samples); beta.fill(0.00);
arma::mat eta(p_q, mcmc_samples); eta.fill(0.00);
arma::vec radius(mcmc_samples); radius.fill(0.00);
arma::mat theta(n_ind, mcmc_samples); theta.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);
arma::mat log_density;

arma::vec off_set(n_ind); off_set.fill(0.00);
if(offset.isNotNull()){
  off_set = Rcpp::as<arma::vec>(offset);
  }

arma::vec tri_als(n_ind); tri_als.fill(1);
if(trials.isNotNull()){
  tri_als = Rcpp::as<arma::vec>(trials);
  }

arma::vec v_index(n_ind); v_index.fill(0);
arma::mat v_exposure_dists(n_ind, m);
for(int j = 0; j < n_ind; ++j){
  
  v_index(j) = v(j) - 1;
  v_exposure_dists.row(j) = exposure_dists.row(v_index(j));
  
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
theta.col(0) = q*eta.col(0);

//Determine Max Possible Exposure
arma::mat radius_max_mat(n_ind, m); radius_max_mat.fill(radius_range(1));
arma::umat comparison_max = (v_exposure_dists < radius_max_mat);
arma::mat numeric_max_mat = arma::conv_to<arma::mat>::from(comparison_max);
arma::vec exposure_max = arma::sum(numeric_max_mat,
                                   1);
double m_max = max(exposure_max);
if(exposure_definition_indicator == 2){
  m_max = 1;  
  }

//Cumulative Counts
if(exposure_definition_indicator == 0){
  
  arma::umat comparison = (v_exposure_dists < radius_mat);
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
  arma::umat comparison = (v_exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  arma::mat prod = corrs%numeric_mat;
  exposure = arma::sum(prod,
                       1);
  exposure = exposure/m_max;
  
  }

//Presence/Absence
if(exposure_definition_indicator == 2){
  
  arma::umat comparison = (v_exposure_dists < radius_mat);
  arma::mat numeric_mat = arma::conv_to<arma::mat>::from(comparison);
  exposure = arma::max(numeric_mat,
                       1);
  
  }

arma::mat Z(n_ind, p_q);
for(int j = 0; j < p_q; ++j){
  Z.col(j) = exposure%q.col(j);
  }

Rcpp::List fit_info = neg_two_loglike_update(y,
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

neg_two_loglike(0) = Rcpp::as<double>(fit_info[0]);
if(waic_info_ind == 1){
  
  log_density = arma::mat(n_ind, mcmc_samples); log_density.fill(0.00);
  log_density.col(0) = Rcpp::as<arma::vec>(fit_info[1]);
  
  }

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
                           p_q,
                           sigma2_eta,
                           omega,
                           lambda,
                           beta.col(j),
                           Z);
   theta.col(j) = q*eta.col(j);
   
   //radius Update
   Rcpp::List radius_output = radius_update(radius_range,
                                            exposure_definition_indicator,
                                            v_exposure_dists,
                                            p_q,
                                            n_ind,
                                            m,
                                            m_max,
                                            x,
                                            q,
                                            off_set,
                                            omega,
                                            lambda,
                                            beta.col(j),
                                            eta.col(j),
                                            radius(j-1),
                                            radius_trans,
                                            exposure,
                                            Z,
                                            metrop_var_radius,
                                            acctot_radius);
   
   radius(j) = Rcpp::as<double>(radius_output[0]);
   acctot_radius = Rcpp::as<int>(radius_output[1]);
   radius_trans = Rcpp::as<double>(radius_output[2]);
   exposure = Rcpp::as<arma::vec>(radius_output[3]);
   Z = Rcpp::as<arma::mat>(radius_output[4]);
   
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
   fit_info = neg_two_loglike_update(y,
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
   
   neg_two_loglike(j) = Rcpp::as<double>(fit_info[0]);
   if(waic_info_ind == 1){
     log_density.col(j) = Rcpp::as<arma::vec>(fit_info[1]);
     }
  
  //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     
     double accrate_radius = round(100*(acctot_radius/(double)j));
     Rcpp::Rcout << "radius Acceptance: " << accrate_radius << "%" << std::endl;
     
     Rcpp::Rcout << "**********************" << std::endl;
    
     }
   
   }

if(waic_info_ind == 0){
  return Rcpp::List::create(Rcpp::Named("exposure_scale") = m_max,
                            Rcpp::Named("r") = r,
                            Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("eta") = eta,
                            Rcpp::Named("radius") = radius,
                            Rcpp::Named("theta") = theta,
                            Rcpp::Named("neg_two_loglike") = neg_two_loglike);
  }

if(waic_info_ind == 1){
  return Rcpp::List::create(Rcpp::Named("exposure_scale") = m_max,
                            Rcpp::Named("r") = r,
                            Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("eta") = eta,
                            Rcpp::Named("radius") = radius,
                            Rcpp::Named("theta") = theta,
                            Rcpp::Named("neg_two_loglike") = neg_two_loglike,
                            Rcpp::Named("log_density") = log_density);
  }

return R_NilValue;
  
}