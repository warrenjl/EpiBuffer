#ifndef __EpiBuffer__
#define __EpiBuffer__

arma::vec rcpp_pgdraw(arma::vec b, 
                      arma::vec c);

Rcpp::NumericVector sampleRcpp(Rcpp::NumericVector x,
                               int size,
                               bool replace,
                               Rcpp::NumericVector prob = Rcpp::NumericVector::create());

Rcpp::List spatial_corr_fun(double phi,
                            arma::mat spatial_dists);

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
                         int m_max,
                         arma::mat x,
                         arma::vec off_set,
                         arma::vec omega,
                         arma::vec lambda,
                         arma::vec beta,
                         arma::vec eta,
                         double radius_old,
                         arma::vec theta,
                         double radius_trans_old,
                         arma::vec poly,
                         arma::vec exposure,
                         arma::mat Z,
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

Rcpp::List gamma_update(arma::vec radius_range,
                        int exposure_definition_indicator,
                        arma::mat v_exposure_dists,
                        int p_d,
                        int n_ind,
                        int m,
                        int m_max,
                        int p_w,
                        arma::mat x,
                        arma::mat v_w,
                        arma::mat v,
                        arma::vec off_set,
                        double sigma2_gamma,
                        arma::vec omega,
                        arma::vec lambda,
                        arma::vec beta, 
                        arma::vec eta,
                        arma::vec gamma_old,
                        arma::vec radius,
                        arma::vec theta,
                        arma::vec radius_trans,
                        arma::vec phi_tilde,
                        arma::mat poly,
                        arma::vec exposure,
                        arma::mat Z,
                        arma::vec metrop_var_gamma,
                        arma::vec acctot_gamma);

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
                           arma::vec acctot_phi_star);

Rcpp::List rho_phi_update(arma::vec radius_range,
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
                          arma::mat dists12,
                          arma::mat dists22,
                          double a_rho_phi,
                          double b_rho_phi,
                          arma::vec omega,
                          arma::vec lambda,
                          arma::vec beta, 
                          arma::vec eta,
                          arma::vec gamma,
                          arma::vec radius,
                          arma::vec theta,
                          double rho_phi_old,
                          arma::vec radius_trans,
                          arma::vec phi_star,
                          arma::vec phi_tilde,
                          Rcpp::List phi_star_corr_info,
                          arma::mat C,
                          arma::mat poly,
                          arma::vec exposure,
                          arma::mat Z,
                          double metrop_var_rho_phi,
                          int acctot_rho_phi);

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

Rcpp::List SpatialBuffers(int mcmc_samples,
                          arma::vec y,
                          arma::mat x,
                          arma::mat w,
                          arma::mat V,
                          arma::vec radius_range,
                          int exposure_definition_indicator,
                          arma::mat exposure_dists,
                          int p_d,
                          arma::mat full_dists,
                          arma::vec metrop_var_gamma,
                          arma::vec metrop_var_phi_star,
                          double metrop_var_rho_phi,
                          int likelihood_indicator,
                          Rcpp::Nullable<Rcpp::NumericVector> offset,
                          Rcpp::Nullable<Rcpp::NumericVector> trials,
                          Rcpp::Nullable<double> a_r_prior,
                          Rcpp::Nullable<double> b_r_prior,
                          Rcpp::Nullable<double> a_sigma2_epsilon_prior,
                          Rcpp::Nullable<double> b_sigma2_epsilon_prior,
                          Rcpp::Nullable<double> sigma2_beta_prior,
                          Rcpp::Nullable<double> sigma2_eta_prior,
                          Rcpp::Nullable<double> sigma2_gamma_prior,
                          Rcpp::Nullable<double> a_rho_phi_prior,
                          Rcpp::Nullable<double> b_rho_phi_prior,
                          Rcpp::Nullable<double> r_init,
                          Rcpp::Nullable<double> sigma2_epsilon_init,
                          Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                          Rcpp::Nullable<Rcpp::NumericVector> eta_init,
                          Rcpp::Nullable<Rcpp::NumericVector> gamma_init,
                          Rcpp::Nullable<double> rho_phi_init);

#endif // __EpiBuffer__
