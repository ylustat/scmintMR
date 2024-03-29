// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// get_opts
List get_opts(int L, Nullable<NumericVector> a_gamma, Nullable<NumericVector> b_gamma, Nullable<NumericVector> a_alpha, Nullable<NumericVector> b_alpha, Nullable<NumericVector> a_beta, Nullable<NumericVector> b_beta, Nullable<double> a, Nullable<double> b, Nullable<int> maxIter, Nullable<int> thin, Nullable<int> burnin);
RcppExport SEXP _scmintMR_get_opts(SEXP LSEXP, SEXP a_gammaSEXP, SEXP b_gammaSEXP, SEXP a_alphaSEXP, SEXP b_alphaSEXP, SEXP a_betaSEXP, SEXP b_betaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP maxIterSEXP, SEXP thinSEXP, SEXP burninSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type L(LSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a_gamma(a_gammaSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type b_gamma(b_gammaSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a_alpha(a_alphaSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type b_alpha(b_alphaSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type a_beta(a_betaSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type b_beta(b_betaSEXP);
    Rcpp::traits::input_parameter< Nullable<double> >::type a(aSEXP);
    Rcpp::traits::input_parameter< Nullable<double> >::type b(bSEXP);
    Rcpp::traits::input_parameter< Nullable<int> >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< Nullable<int> >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< Nullable<int> >::type burnin(burninSEXP);
    rcpp_result_gen = Rcpp::wrap(get_opts(L, a_gamma, b_gamma, a_alpha, b_alpha, a_beta, b_beta, a, b, maxIter, thin, burnin));
    return rcpp_result_gen;
END_RCPP
}
// check_missing
arma::mat check_missing(List gammah);
RcppExport SEXP _scmintMR_check_missing(SEXP gammahSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type gammah(gammahSEXP);
    rcpp_result_gen = Rcpp::wrap(check_missing(gammah));
    return rcpp_result_gen;
END_RCPP
}
// generate_transformation_indicator
arma::mat generate_transformation_indicator(const arma::mat gammah_elem);
RcppExport SEXP _scmintMR_generate_transformation_indicator(SEXP gammah_elemSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type gammah_elem(gammah_elemSEXP);
    rcpp_result_gen = Rcpp::wrap(generate_transformation_indicator(gammah_elem));
    return rcpp_result_gen;
END_RCPP
}
// keep_nonmissing_column
arma::mat keep_nonmissing_column(const arma::mat X);
RcppExport SEXP _scmintMR_keep_nonmissing_column(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(keep_nonmissing_column(X));
    return rcpp_result_gen;
END_RCPP
}
// spca
List spca(const arma::mat x, const arma::mat y);
RcppExport SEXP _scmintMR_spca(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(spca(x, y));
    return rcpp_result_gen;
END_RCPP
}
// VarCompTest_cpp
List VarCompTest_cpp(const arma::vec y, const arma::mat Z);
RcppExport SEXP _scmintMR_VarCompTest_cpp(SEXP ySEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(VarCompTest_cpp(y, Z));
    return rcpp_result_gen;
END_RCPP
}
// cppmissForest
arma::mat cppmissForest(arma::mat gammah);
RcppExport SEXP _scmintMR_cppmissForest(SEXP gammahSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type gammah(gammahSEXP);
    rcpp_result_gen = Rcpp::wrap(cppmissForest(gammah));
    return rcpp_result_gen;
END_RCPP
}
// mintMR_Impute_MVL
List mintMR_Impute_MVL(List gammah, const List Gammah, List se1, const List se2, const List corr_mat, const List group, const List opts, const arma::mat Lambda, bool display_progress, int latent_dim, int CC, int PC1, int PC2, String missing_method, String mvl_method, int epochs, bool fast_impute);
RcppExport SEXP _scmintMR_mintMR_Impute_MVL(SEXP gammahSEXP, SEXP GammahSEXP, SEXP se1SEXP, SEXP se2SEXP, SEXP corr_matSEXP, SEXP groupSEXP, SEXP optsSEXP, SEXP LambdaSEXP, SEXP display_progressSEXP, SEXP latent_dimSEXP, SEXP CCSEXP, SEXP PC1SEXP, SEXP PC2SEXP, SEXP missing_methodSEXP, SEXP mvl_methodSEXP, SEXP epochsSEXP, SEXP fast_imputeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type gammah(gammahSEXP);
    Rcpp::traits::input_parameter< const List >::type Gammah(GammahSEXP);
    Rcpp::traits::input_parameter< List >::type se1(se1SEXP);
    Rcpp::traits::input_parameter< const List >::type se2(se2SEXP);
    Rcpp::traits::input_parameter< const List >::type corr_mat(corr_matSEXP);
    Rcpp::traits::input_parameter< const List >::type group(groupSEXP);
    Rcpp::traits::input_parameter< const List >::type opts(optsSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type Lambda(LambdaSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    Rcpp::traits::input_parameter< int >::type latent_dim(latent_dimSEXP);
    Rcpp::traits::input_parameter< int >::type CC(CCSEXP);
    Rcpp::traits::input_parameter< int >::type PC1(PC1SEXP);
    Rcpp::traits::input_parameter< int >::type PC2(PC2SEXP);
    Rcpp::traits::input_parameter< String >::type missing_method(missing_methodSEXP);
    Rcpp::traits::input_parameter< String >::type mvl_method(mvl_methodSEXP);
    Rcpp::traits::input_parameter< int >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type fast_impute(fast_imputeSEXP);
    rcpp_result_gen = Rcpp::wrap(mintMR_Impute_MVL(gammah, Gammah, se1, se2, corr_mat, group, opts, Lambda, display_progress, latent_dim, CC, PC1, PC2, missing_method, mvl_method, epochs, fast_impute));
    return rcpp_result_gen;
END_RCPP
}
// mintMR
List mintMR(List gammah, const List Gammah, List se1, const List se2, Nullable<List> group, Nullable<List> opts, Nullable<List> corr_mat, Nullable<arma::mat> reference, Nullable<arma::mat> Lambda, int CC, int PC1, int PC2, int latent_dim, bool display_progress, String missing_method, String mvl_method, int epochs, bool fast_impute, bool deep);
RcppExport SEXP _scmintMR_mintMR(SEXP gammahSEXP, SEXP GammahSEXP, SEXP se1SEXP, SEXP se2SEXP, SEXP groupSEXP, SEXP optsSEXP, SEXP corr_matSEXP, SEXP referenceSEXP, SEXP LambdaSEXP, SEXP CCSEXP, SEXP PC1SEXP, SEXP PC2SEXP, SEXP latent_dimSEXP, SEXP display_progressSEXP, SEXP missing_methodSEXP, SEXP mvl_methodSEXP, SEXP epochsSEXP, SEXP fast_imputeSEXP, SEXP deepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type gammah(gammahSEXP);
    Rcpp::traits::input_parameter< const List >::type Gammah(GammahSEXP);
    Rcpp::traits::input_parameter< List >::type se1(se1SEXP);
    Rcpp::traits::input_parameter< const List >::type se2(se2SEXP);
    Rcpp::traits::input_parameter< Nullable<List> >::type group(groupSEXP);
    Rcpp::traits::input_parameter< Nullable<List> >::type opts(optsSEXP);
    Rcpp::traits::input_parameter< Nullable<List> >::type corr_mat(corr_matSEXP);
    Rcpp::traits::input_parameter< Nullable<arma::mat> >::type reference(referenceSEXP);
    Rcpp::traits::input_parameter< Nullable<arma::mat> >::type Lambda(LambdaSEXP);
    Rcpp::traits::input_parameter< int >::type CC(CCSEXP);
    Rcpp::traits::input_parameter< int >::type PC1(PC1SEXP);
    Rcpp::traits::input_parameter< int >::type PC2(PC2SEXP);
    Rcpp::traits::input_parameter< int >::type latent_dim(latent_dimSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    Rcpp::traits::input_parameter< String >::type missing_method(missing_methodSEXP);
    Rcpp::traits::input_parameter< String >::type mvl_method(mvl_methodSEXP);
    Rcpp::traits::input_parameter< int >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< bool >::type fast_impute(fast_imputeSEXP);
    Rcpp::traits::input_parameter< bool >::type deep(deepSEXP);
    rcpp_result_gen = Rcpp::wrap(mintMR(gammah, Gammah, se1, se2, group, opts, corr_mat, reference, Lambda, CC, PC1, PC2, latent_dim, display_progress, missing_method, mvl_method, epochs, fast_impute, deep));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_scmintMR_get_opts", (DL_FUNC) &_scmintMR_get_opts, 12},
    {"_scmintMR_check_missing", (DL_FUNC) &_scmintMR_check_missing, 1},
    {"_scmintMR_generate_transformation_indicator", (DL_FUNC) &_scmintMR_generate_transformation_indicator, 1},
    {"_scmintMR_keep_nonmissing_column", (DL_FUNC) &_scmintMR_keep_nonmissing_column, 1},
    {"_scmintMR_spca", (DL_FUNC) &_scmintMR_spca, 2},
    {"_scmintMR_VarCompTest_cpp", (DL_FUNC) &_scmintMR_VarCompTest_cpp, 2},
    {"_scmintMR_cppmissForest", (DL_FUNC) &_scmintMR_cppmissForest, 1},
    {"_scmintMR_mintMR_Impute_MVL", (DL_FUNC) &_scmintMR_mintMR_Impute_MVL, 17},
    {"_scmintMR_mintMR", (DL_FUNC) &_scmintMR_mintMR, 19},
    {NULL, NULL, 0}
};

RcppExport void R_init_scmintMR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
