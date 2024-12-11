// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// SingleBuffer
Rcpp::List SingleBuffer(int mcmc_samples, arma::vec y, arma::mat x, arma::mat v, arma::vec radius_range, int exposure_definition_indicator, arma::mat exposure_dists, int p_d, double metrop_var_radius, int likelihood_indicator, Rcpp::Nullable<Rcpp::NumericVector> offset, Rcpp::Nullable<Rcpp::NumericVector> trials, Rcpp::Nullable<double> a_r_prior, Rcpp::Nullable<double> b_r_prior, Rcpp::Nullable<double> a_sigma2_epsilon_prior, Rcpp::Nullable<double> b_sigma2_epsilon_prior, Rcpp::Nullable<double> sigma2_beta_prior, Rcpp::Nullable<double> sigma2_eta_prior, Rcpp::Nullable<double> r_init, Rcpp::Nullable<double> sigma2_epsilon_init, Rcpp::Nullable<Rcpp::NumericVector> beta_init, Rcpp::Nullable<Rcpp::NumericVector> eta_init, Rcpp::Nullable<double> radius_init);
RcppExport SEXP _EpiBuffer_SingleBuffer(SEXP mcmc_samplesSEXP, SEXP ySEXP, SEXP xSEXP, SEXP vSEXP, SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP exposure_distsSEXP, SEXP p_dSEXP, SEXP metrop_var_radiusSEXP, SEXP likelihood_indicatorSEXP, SEXP offsetSEXP, SEXP trialsSEXP, SEXP a_r_priorSEXP, SEXP b_r_priorSEXP, SEXP a_sigma2_epsilon_priorSEXP, SEXP b_sigma2_epsilon_priorSEXP, SEXP sigma2_beta_priorSEXP, SEXP sigma2_eta_priorSEXP, SEXP r_initSEXP, SEXP sigma2_epsilon_initSEXP, SEXP beta_initSEXP, SEXP eta_initSEXP, SEXP radius_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type mcmc_samples(mcmc_samplesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type exposure_dists(exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_var_radius(metrop_var_radiusSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type trials(trialsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type a_r_prior(a_r_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type b_r_prior(b_r_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type a_sigma2_epsilon_prior(a_sigma2_epsilon_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type b_sigma2_epsilon_prior(b_sigma2_epsilon_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_beta_prior(sigma2_beta_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_eta_prior(sigma2_eta_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type r_init(r_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_epsilon_init(sigma2_epsilon_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type beta_init(beta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type eta_init(eta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type radius_init(radius_initSEXP);
    rcpp_result_gen = Rcpp::wrap(SingleBuffer(mcmc_samples, y, x, v, radius_range, exposure_definition_indicator, exposure_dists, p_d, metrop_var_radius, likelihood_indicator, offset, trials, a_r_prior, b_r_prior, a_sigma2_epsilon_prior, b_sigma2_epsilon_prior, sigma2_beta_prior, sigma2_eta_prior, r_init, sigma2_epsilon_init, beta_init, eta_init, radius_init));
    return rcpp_result_gen;
END_RCPP
}
// SpatialBuffers
Rcpp::List SpatialBuffers(int mcmc_samples, arma::vec y, arma::mat x, arma::mat w, arma::mat v, arma::vec radius_range, int exposure_definition_indicator, arma::mat exposure_dists, int p_d, arma::mat full_dists, arma::vec metrop_var_gamma, arma::vec metrop_var_phi_star, double metrop_var_rho_phi, int likelihood_indicator, Rcpp::Nullable<Rcpp::NumericVector> offset, Rcpp::Nullable<Rcpp::NumericVector> trials, Rcpp::Nullable<double> a_r_prior, Rcpp::Nullable<double> b_r_prior, Rcpp::Nullable<double> a_sigma2_epsilon_prior, Rcpp::Nullable<double> b_sigma2_epsilon_prior, Rcpp::Nullable<double> sigma2_beta_prior, Rcpp::Nullable<double> sigma2_eta_prior, Rcpp::Nullable<double> sigma2_gamma_prior, Rcpp::Nullable<double> a_rho_phi_prior, Rcpp::Nullable<double> b_rho_phi_prior, Rcpp::Nullable<double> r_init, Rcpp::Nullable<double> sigma2_epsilon_init, Rcpp::Nullable<Rcpp::NumericVector> beta_init, Rcpp::Nullable<Rcpp::NumericVector> eta_init, Rcpp::Nullable<Rcpp::NumericVector> gamma_init, Rcpp::Nullable<double> rho_phi_init);
RcppExport SEXP _EpiBuffer_SpatialBuffers(SEXP mcmc_samplesSEXP, SEXP ySEXP, SEXP xSEXP, SEXP wSEXP, SEXP vSEXP, SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP exposure_distsSEXP, SEXP p_dSEXP, SEXP full_distsSEXP, SEXP metrop_var_gammaSEXP, SEXP metrop_var_phi_starSEXP, SEXP metrop_var_rho_phiSEXP, SEXP likelihood_indicatorSEXP, SEXP offsetSEXP, SEXP trialsSEXP, SEXP a_r_priorSEXP, SEXP b_r_priorSEXP, SEXP a_sigma2_epsilon_priorSEXP, SEXP b_sigma2_epsilon_priorSEXP, SEXP sigma2_beta_priorSEXP, SEXP sigma2_eta_priorSEXP, SEXP sigma2_gamma_priorSEXP, SEXP a_rho_phi_priorSEXP, SEXP b_rho_phi_priorSEXP, SEXP r_initSEXP, SEXP sigma2_epsilon_initSEXP, SEXP beta_initSEXP, SEXP eta_initSEXP, SEXP gamma_initSEXP, SEXP rho_phi_initSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type mcmc_samples(mcmc_samplesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v(vSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type exposure_dists(exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type full_dists(full_distsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type metrop_var_gamma(metrop_var_gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type metrop_var_phi_star(metrop_var_phi_starSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_var_rho_phi(metrop_var_rho_phiSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type trials(trialsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type a_r_prior(a_r_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type b_r_prior(b_r_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type a_sigma2_epsilon_prior(a_sigma2_epsilon_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type b_sigma2_epsilon_prior(b_sigma2_epsilon_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_beta_prior(sigma2_beta_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_eta_prior(sigma2_eta_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_gamma_prior(sigma2_gamma_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type a_rho_phi_prior(a_rho_phi_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type b_rho_phi_prior(b_rho_phi_priorSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type r_init(r_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma2_epsilon_init(sigma2_epsilon_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type beta_init(beta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type eta_init(eta_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type gamma_init(gamma_initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type rho_phi_init(rho_phi_initSEXP);
    rcpp_result_gen = Rcpp::wrap(SpatialBuffers(mcmc_samples, y, x, w, v, radius_range, exposure_definition_indicator, exposure_dists, p_d, full_dists, metrop_var_gamma, metrop_var_phi_star, metrop_var_rho_phi, likelihood_indicator, offset, trials, a_r_prior, b_r_prior, a_sigma2_epsilon_prior, b_sigma2_epsilon_prior, sigma2_beta_prior, sigma2_eta_prior, sigma2_gamma_prior, a_rho_phi_prior, b_rho_phi_prior, r_init, sigma2_epsilon_init, beta_init, eta_init, gamma_init, rho_phi_init));
    return rcpp_result_gen;
END_RCPP
}
// beta_update
arma::vec beta_update(arma::mat x, arma::vec off_set, int n_ind, int p_x, double sigma2_beta, arma::vec omega, arma::vec lambda, arma::vec eta_old, arma::mat Z);
RcppExport SEXP _EpiBuffer_beta_update(SEXP xSEXP, SEXP off_setSEXP, SEXP n_indSEXP, SEXP p_xSEXP, SEXP sigma2_betaSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP eta_oldSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type p_x(p_xSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_beta(sigma2_betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta_old(eta_oldSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(beta_update(x, off_set, n_ind, p_x, sigma2_beta, omega, lambda, eta_old, Z));
    return rcpp_result_gen;
END_RCPP
}
// eta_update
arma::vec eta_update(arma::mat x, arma::vec off_set, int n_ind, int p_d, double sigma2_eta, arma::vec omega, arma::vec lambda, arma::vec beta, arma::mat Z);
RcppExport SEXP _EpiBuffer_eta_update(SEXP xSEXP, SEXP off_setSEXP, SEXP n_indSEXP, SEXP p_dSEXP, SEXP sigma2_etaSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP betaSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_eta(sigma2_etaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(eta_update(x, off_set, n_ind, p_d, sigma2_eta, omega, lambda, beta, Z));
    return rcpp_result_gen;
END_RCPP
}
// gamma_update
Rcpp::List gamma_update(arma::vec radius_range, int exposure_definition_indicator, arma::mat v_exposure_dists, int p_d, int n_ind, int m, int m_max, int p_w, arma::mat x, arma::mat v_w, arma::vec v_index, arma::vec off_set, double sigma2_gamma, arma::vec omega, arma::vec lambda, arma::vec beta, arma::vec eta, arma::vec gamma_old, arma::vec radius, arma::vec theta, arma::vec radius_trans, arma::vec phi_tilde, arma::mat poly, arma::vec exposure, arma::mat Z, arma::vec metrop_var_gamma, arma::vec acctot_gamma);
RcppExport SEXP _EpiBuffer_gamma_update(SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP v_exposure_distsSEXP, SEXP p_dSEXP, SEXP n_indSEXP, SEXP mSEXP, SEXP m_maxSEXP, SEXP p_wSEXP, SEXP xSEXP, SEXP v_wSEXP, SEXP v_indexSEXP, SEXP off_setSEXP, SEXP sigma2_gammaSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP gamma_oldSEXP, SEXP radiusSEXP, SEXP thetaSEXP, SEXP radius_transSEXP, SEXP phi_tildeSEXP, SEXP polySEXP, SEXP exposureSEXP, SEXP ZSEXP, SEXP metrop_var_gammaSEXP, SEXP acctot_gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_exposure_dists(v_exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< int >::type p_w(p_wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_w(v_wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type v_index(v_indexSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_gamma(sigma2_gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma_old(gamma_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius_trans(radius_transSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phi_tilde(phi_tildeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type poly(polySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type exposure(exposureSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type metrop_var_gamma(metrop_var_gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type acctot_gamma(acctot_gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(gamma_update(radius_range, exposure_definition_indicator, v_exposure_dists, p_d, n_ind, m, m_max, p_w, x, v_w, v_index, off_set, sigma2_gamma, omega, lambda, beta, eta, gamma_old, radius, theta, radius_trans, phi_tilde, poly, exposure, Z, metrop_var_gamma, acctot_gamma));
    return rcpp_result_gen;
END_RCPP
}
// latent_update
Rcpp::List latent_update(arma::vec y, arma::mat x, arma::vec off_set, arma::vec tri_als, int likelihood_indicator, int n_ind, int r_old, arma::vec beta_old, arma::vec eta_old, arma::mat Z);
RcppExport SEXP _EpiBuffer_latent_update(SEXP ySEXP, SEXP xSEXP, SEXP off_setSEXP, SEXP tri_alsSEXP, SEXP likelihood_indicatorSEXP, SEXP n_indSEXP, SEXP r_oldSEXP, SEXP beta_oldSEXP, SEXP eta_oldSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tri_als(tri_alsSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type r_old(r_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_old(beta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta_old(eta_oldSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(latent_update(y, x, off_set, tri_als, likelihood_indicator, n_ind, r_old, beta_old, eta_old, Z));
    return rcpp_result_gen;
END_RCPP
}
// neg_two_loglike_update
double neg_two_loglike_update(arma::vec y, arma::mat x, arma::vec off_set, arma::vec tri_als, int likelihood_indicator, int n_ind, int r, double sigma2_epsilon, arma::vec beta, arma::vec eta, arma::mat Z);
RcppExport SEXP _EpiBuffer_neg_two_loglike_update(SEXP ySEXP, SEXP xSEXP, SEXP off_setSEXP, SEXP tri_alsSEXP, SEXP likelihood_indicatorSEXP, SEXP n_indSEXP, SEXP rSEXP, SEXP sigma2_epsilonSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tri_als(tri_alsSEXP);
    Rcpp::traits::input_parameter< int >::type likelihood_indicator(likelihood_indicatorSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type sigma2_epsilon(sigma2_epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(neg_two_loglike_update(y, x, off_set, tri_als, likelihood_indicator, n_ind, r, sigma2_epsilon, beta, eta, Z));
    return rcpp_result_gen;
END_RCPP
}
// phi_star_update
Rcpp::List phi_star_update(arma::vec radius_range, int exposure_definition_indicator, arma::mat v_exposure_dists, int p_d, int n_ind, int n_grid, int m, int m_max, int p_w, arma::mat x, arma::mat v_w, arma::vec v_index, arma::vec off_set, arma::vec omega, arma::vec lambda, arma::vec beta, arma::vec eta, arma::vec gamma, arma::vec radius, arma::vec theta, arma::vec radius_trans, arma::vec phi_star, arma::vec phi_tilde, arma::mat phi_star_corr_inv, arma::mat C, arma::mat poly, arma::vec exposure, arma::mat Z, arma::vec metrop_var_phi_star, arma::vec acctot_phi_star);
RcppExport SEXP _EpiBuffer_phi_star_update(SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP v_exposure_distsSEXP, SEXP p_dSEXP, SEXP n_indSEXP, SEXP n_gridSEXP, SEXP mSEXP, SEXP m_maxSEXP, SEXP p_wSEXP, SEXP xSEXP, SEXP v_wSEXP, SEXP v_indexSEXP, SEXP off_setSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP gammaSEXP, SEXP radiusSEXP, SEXP thetaSEXP, SEXP radius_transSEXP, SEXP phi_starSEXP, SEXP phi_tildeSEXP, SEXP phi_star_corr_invSEXP, SEXP CSEXP, SEXP polySEXP, SEXP exposureSEXP, SEXP ZSEXP, SEXP metrop_var_phi_starSEXP, SEXP acctot_phi_starSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_exposure_dists(v_exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type n_grid(n_gridSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< int >::type p_w(p_wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_w(v_wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type v_index(v_indexSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius_trans(radius_transSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phi_star(phi_starSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phi_tilde(phi_tildeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type phi_star_corr_inv(phi_star_corr_invSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type poly(polySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type exposure(exposureSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type metrop_var_phi_star(metrop_var_phi_starSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type acctot_phi_star(acctot_phi_starSEXP);
    rcpp_result_gen = Rcpp::wrap(phi_star_update(radius_range, exposure_definition_indicator, v_exposure_dists, p_d, n_ind, n_grid, m, m_max, p_w, x, v_w, v_index, off_set, omega, lambda, beta, eta, gamma, radius, theta, radius_trans, phi_star, phi_tilde, phi_star_corr_inv, C, poly, exposure, Z, metrop_var_phi_star, acctot_phi_star));
    return rcpp_result_gen;
END_RCPP
}
// r_update
int r_update(arma::vec y, arma::mat x, arma::vec off_set, int n_ind, int a_r, int b_r, arma::vec beta, arma::vec eta, arma::mat Z);
RcppExport SEXP _EpiBuffer_r_update(SEXP ySEXP, SEXP xSEXP, SEXP off_setSEXP, SEXP n_indSEXP, SEXP a_rSEXP, SEXP b_rSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type a_r(a_rSEXP);
    Rcpp::traits::input_parameter< int >::type b_r(b_rSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(r_update(y, x, off_set, n_ind, a_r, b_r, beta, eta, Z));
    return rcpp_result_gen;
END_RCPP
}
// radius_update
Rcpp::List radius_update(arma::vec radius_range, int exposure_definition_indicator, arma::mat v_exposure_dists, int p_d, int n_ind, int m, int m_max, arma::mat x, arma::vec off_set, arma::vec omega, arma::vec lambda, arma::vec beta, arma::vec eta, double radius_old, arma::vec theta, double radius_trans_old, arma::vec poly, arma::vec exposure, arma::mat Z, double metrop_var_radius, int acctot_radius);
RcppExport SEXP _EpiBuffer_radius_update(SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP v_exposure_distsSEXP, SEXP p_dSEXP, SEXP n_indSEXP, SEXP mSEXP, SEXP m_maxSEXP, SEXP xSEXP, SEXP off_setSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP radius_oldSEXP, SEXP thetaSEXP, SEXP radius_trans_oldSEXP, SEXP polySEXP, SEXP exposureSEXP, SEXP ZSEXP, SEXP metrop_var_radiusSEXP, SEXP acctot_radiusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_exposure_dists(v_exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< double >::type radius_old(radius_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type radius_trans_old(radius_trans_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type poly(polySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type exposure(exposureSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_var_radius(metrop_var_radiusSEXP);
    Rcpp::traits::input_parameter< int >::type acctot_radius(acctot_radiusSEXP);
    rcpp_result_gen = Rcpp::wrap(radius_update(radius_range, exposure_definition_indicator, v_exposure_dists, p_d, n_ind, m, m_max, x, off_set, omega, lambda, beta, eta, radius_old, theta, radius_trans_old, poly, exposure, Z, metrop_var_radius, acctot_radius));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_pgdraw
arma::vec rcpp_pgdraw(arma::vec b, arma::vec c);
RcppExport SEXP _EpiBuffer_rcpp_pgdraw(SEXP bSEXP, SEXP cSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type b(bSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c(cSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_pgdraw(b, c));
    return rcpp_result_gen;
END_RCPP
}
// rho_phi_update
Rcpp::List rho_phi_update(arma::vec radius_range, int exposure_definition_indicator, arma::mat v_exposure_dists, int p_d, int n_ind, int n_grid, int m, int m_max, int p_w, arma::mat x, arma::mat v_w, arma::vec v_index, arma::vec off_set, arma::mat dists12, arma::mat dists22, double a_rho_phi, double b_rho_phi, arma::vec omega, arma::vec lambda, arma::vec beta, arma::vec eta, arma::vec gamma, arma::vec radius, arma::vec theta, double rho_phi_old, arma::vec radius_trans, arma::vec phi_star, arma::vec phi_tilde, Rcpp::List phi_star_corr_info, arma::mat C, arma::mat poly, arma::vec exposure, arma::mat Z, double metrop_var_rho_phi, int acctot_rho_phi);
RcppExport SEXP _EpiBuffer_rho_phi_update(SEXP radius_rangeSEXP, SEXP exposure_definition_indicatorSEXP, SEXP v_exposure_distsSEXP, SEXP p_dSEXP, SEXP n_indSEXP, SEXP n_gridSEXP, SEXP mSEXP, SEXP m_maxSEXP, SEXP p_wSEXP, SEXP xSEXP, SEXP v_wSEXP, SEXP v_indexSEXP, SEXP off_setSEXP, SEXP dists12SEXP, SEXP dists22SEXP, SEXP a_rho_phiSEXP, SEXP b_rho_phiSEXP, SEXP omegaSEXP, SEXP lambdaSEXP, SEXP betaSEXP, SEXP etaSEXP, SEXP gammaSEXP, SEXP radiusSEXP, SEXP thetaSEXP, SEXP rho_phi_oldSEXP, SEXP radius_transSEXP, SEXP phi_starSEXP, SEXP phi_tildeSEXP, SEXP phi_star_corr_infoSEXP, SEXP CSEXP, SEXP polySEXP, SEXP exposureSEXP, SEXP ZSEXP, SEXP metrop_var_rho_phiSEXP, SEXP acctot_rho_phiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type radius_range(radius_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type exposure_definition_indicator(exposure_definition_indicatorSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_exposure_dists(v_exposure_distsSEXP);
    Rcpp::traits::input_parameter< int >::type p_d(p_dSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< int >::type n_grid(n_gridSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type m_max(m_maxSEXP);
    Rcpp::traits::input_parameter< int >::type p_w(p_wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type v_w(v_wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type v_index(v_indexSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dists12(dists12SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dists22(dists22SEXP);
    Rcpp::traits::input_parameter< double >::type a_rho_phi(a_rho_phiSEXP);
    Rcpp::traits::input_parameter< double >::type b_rho_phi(b_rho_phiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius(radiusSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type rho_phi_old(rho_phi_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type radius_trans(radius_transSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phi_star(phi_starSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type phi_tilde(phi_tildeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type phi_star_corr_info(phi_star_corr_infoSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type poly(polySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type exposure(exposureSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_var_rho_phi(metrop_var_rho_phiSEXP);
    Rcpp::traits::input_parameter< int >::type acctot_rho_phi(acctot_rho_phiSEXP);
    rcpp_result_gen = Rcpp::wrap(rho_phi_update(radius_range, exposure_definition_indicator, v_exposure_dists, p_d, n_ind, n_grid, m, m_max, p_w, x, v_w, v_index, off_set, dists12, dists22, a_rho_phi, b_rho_phi, omega, lambda, beta, eta, gamma, radius, theta, rho_phi_old, radius_trans, phi_star, phi_tilde, phi_star_corr_info, C, poly, exposure, Z, metrop_var_rho_phi, acctot_rho_phi));
    return rcpp_result_gen;
END_RCPP
}
// sigma2_epsilon_update
double sigma2_epsilon_update(arma::vec y, arma::mat x, arma::vec off_set, int n_ind, double a_sigma2_epsilon, double b_sigma2_epsilon, arma::vec beta_old, arma::vec eta_old, arma::mat Z);
RcppExport SEXP _EpiBuffer_sigma2_epsilon_update(SEXP ySEXP, SEXP xSEXP, SEXP off_setSEXP, SEXP n_indSEXP, SEXP a_sigma2_epsilonSEXP, SEXP b_sigma2_epsilonSEXP, SEXP beta_oldSEXP, SEXP eta_oldSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type off_set(off_setSEXP);
    Rcpp::traits::input_parameter< int >::type n_ind(n_indSEXP);
    Rcpp::traits::input_parameter< double >::type a_sigma2_epsilon(a_sigma2_epsilonSEXP);
    Rcpp::traits::input_parameter< double >::type b_sigma2_epsilon(b_sigma2_epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta_old(beta_oldSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta_old(eta_oldSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(sigma2_epsilon_update(y, x, off_set, n_ind, a_sigma2_epsilon, b_sigma2_epsilon, beta_old, eta_old, Z));
    return rcpp_result_gen;
END_RCPP
}
// spatial_corr_fun
Rcpp::List spatial_corr_fun(double phi, arma::mat spatial_dists);
RcppExport SEXP _EpiBuffer_spatial_corr_fun(SEXP phiSEXP, SEXP spatial_distsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type spatial_dists(spatial_distsSEXP);
    rcpp_result_gen = Rcpp::wrap(spatial_corr_fun(phi, spatial_dists));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_EpiBuffer_SingleBuffer", (DL_FUNC) &_EpiBuffer_SingleBuffer, 23},
    {"_EpiBuffer_SpatialBuffers", (DL_FUNC) &_EpiBuffer_SpatialBuffers, 31},
    {"_EpiBuffer_beta_update", (DL_FUNC) &_EpiBuffer_beta_update, 9},
    {"_EpiBuffer_eta_update", (DL_FUNC) &_EpiBuffer_eta_update, 9},
    {"_EpiBuffer_gamma_update", (DL_FUNC) &_EpiBuffer_gamma_update, 27},
    {"_EpiBuffer_latent_update", (DL_FUNC) &_EpiBuffer_latent_update, 10},
    {"_EpiBuffer_neg_two_loglike_update", (DL_FUNC) &_EpiBuffer_neg_two_loglike_update, 11},
    {"_EpiBuffer_phi_star_update", (DL_FUNC) &_EpiBuffer_phi_star_update, 30},
    {"_EpiBuffer_r_update", (DL_FUNC) &_EpiBuffer_r_update, 9},
    {"_EpiBuffer_radius_update", (DL_FUNC) &_EpiBuffer_radius_update, 21},
    {"_EpiBuffer_rcpp_pgdraw", (DL_FUNC) &_EpiBuffer_rcpp_pgdraw, 2},
    {"_EpiBuffer_rho_phi_update", (DL_FUNC) &_EpiBuffer_rho_phi_update, 35},
    {"_EpiBuffer_sigma2_epsilon_update", (DL_FUNC) &_EpiBuffer_sigma2_epsilon_update, 9},
    {"_EpiBuffer_spatial_corr_fun", (DL_FUNC) &_EpiBuffer_spatial_corr_fun, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_EpiBuffer(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
