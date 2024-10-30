#ifndef __EpiBuffer__
#define __EpiBuffer__

arma::vec rcpp_pgdraw(arma::vec b, 
                      arma::vec c);

Rcpp::NumericVector sampleRcpp(Rcpp::NumericVector x,
                               int size,
                               bool replace,
                               Rcpp::NumericVector prob = Rcpp::NumericVector::create());

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              arma::vec off_set,
                              arma::vec tri_als,
                              int likelihood_indicator,
                              int n_ind,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,
                              arma::vec eta,
                              arma::mat Z);

Rcpp::List latent_update(arma::vec y,
                         arma::mat x,
                         arma::vec off_set,
                         arma::vec tri_als,
                         int likelihood_indicator,
                         int n_ind,
                         int r_old,
                         arma::vec beta_old,
                         arma::vec eta_old,
                         arma::mat Z);

double sigma2_epsilon_update(arma::vec y,
                             arma::mat x,
                             arma::vec off_set,
                             int n_ind,
                             double a_sigma2_epsilon,
                             double b_sigma2_epsilon,
                             arma::vec beta_old,
                             arma::vec eta_old,
                             arma::mat Z);

arma::vec beta_update(arma::mat x,
                      arma::vec off_set,
                      int n_ind,
                      int p_x,
                      double sigma2_beta,
                      arma::vec omega,
                      arma::vec lambda,
                      arma::vec eta_old,
                      arma::mat Z);

arma::vec eta_update(arma::mat x,
                     arma::vec off_set,
                     int n_ind,
                     int p_d,
                     double sigma2_eta,
                     arma::vec omega,
                     arma::vec lambda,
                     arma::vec beta,
                     arma::mat Z);

Rcpp::List radius_update(arma::vec radius_range,
                         int exposure_definition_indicator,
                         arma::mat exposure_dists,
                         int p_d,
                         int n_ind,
                         int m,
                         arma::mat x,
                         arma::vec off_set,
                         arma::vec omega,
                         arma::vec lambda,
                         arma::vec beta,
                         arma::vec eta,
                         double radius_trans_old,
                         double radius_old,
                         arma::vec poly,
                         arma::vec exposure,
                         arma::mat Z,
                         arma::vec theta_keep,
                         double metrop_var_radius,
                         int acctot_radius);

int r_update(arma::vec y,
             arma::mat x,
             arma::vec off_set,
             int n_ind,
             int a_r,
             int b_r,
             arma::vec beta,
             arma::vec eta_keep,
             arma::mat Z);

Rcpp::List SingleBuffer(int mcmc_samples,
                        arma::vec y,
                        arma::mat x,
                        arma::vec radius_range,
                        int exposure_definition_indicator,
                        arma::mat exposure_dists,
                        int p_d,
                        double metrop_var_radius,
                        int likelihood_indicator,
                        Rcpp::Nullable<Rcpp::NumericVector> offset,
                        Rcpp::Nullable<Rcpp::NumericVector> trials,
                        Rcpp::Nullable<double> a_r_prior,
                        Rcpp::Nullable<double> b_r_prior,
                        Rcpp::Nullable<double> a_sigma2_epsilon_prior,
                        Rcpp::Nullable<double> b_sigma2_epsilon_prior,
                        Rcpp::Nullable<double> sigma2_beta_prior,
                        Rcpp::Nullable<double> sigma2_eta_prior,
                        Rcpp::Nullable<double> r_init,
                        Rcpp::Nullable<double> sigma2_epsilon_init,
                        Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                        Rcpp::Nullable<Rcpp::NumericVector> eta_init,
                        Rcpp::Nullable<double> radius_init);

#endif // __EpiBuffer__
