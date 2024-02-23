#include <RcppArmadillo.h>
#include <random>
#include <math.h>
#include <Rcpp.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <ctime>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;
// Define the function

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>

Environment pkg1 = Environment::namespace_env("base");
Function eigen = pkg1["eigen"];

Environment pkg2 = Environment::namespace_env("SKAT");
Function Get_Davies_PVal = pkg2["Get_Davies_PVal"];

Environment pkg = Environment::namespace_env("CCA");
Function cc = pkg["cc"];

Environment pkgImp = Environment::namespace_env("rMIDAS");
Function convert = pkgImp["convert"];
Function train = pkgImp["train"];
Function complete = pkgImp["complete"];

Environment pkgMF = Environment::namespace_env("missForest");
Function missForest = pkgMF["missForest"];

Environment pkgBase = Environment::namespace_env("base");
Function dataframe = pkgBase["data.frame"];
Function datamatrix = pkgBase["data.matrix"];

Environment pkgMVL = Environment::namespace_env("MVL");
Function DeepCCA = pkgMVL["DeepCCA"];
Function MIDAS = pkgMVL["MIDAS"];
Function fillNA = pkgMVL["fillNA"];
Function fillmean = pkgMVL["fillmean"];
Function vc_test = pkgMVL["variance_component_test"];


arma::mat normalize_mat(arma::mat X) {
  int p = X.n_cols;
  
  mat colmean = mean(X, 0);
  X.each_row() -= colmean;
  
  mat stddevs = stddev(X, 0, 0);
  for (int i = 0; i < p; i++) {
    if (stddevs(i) != 0) {
      X.col(i) /= stddevs[i];
    }
  }
  
  return X;
}

arma::mat do_call_rbind_vecs(List list_of_matrices) {
  int num_matrices = list_of_matrices.size();
  mat combined_matrix;
  
  for (int i = 0; i < num_matrices; i++) {
    NumericMatrix current_matrix = list_of_matrices[i];
    mat current_matrix_arma = as<mat>(current_matrix);
    combined_matrix = join_rows(combined_matrix, current_matrix_arma);
  }
  
  return trans(combined_matrix); // need transpose
}

arma::mat do_call_cbind_vecs(List list_of_matrices) {
  int num_matrices = list_of_matrices.size();
  mat combined_matrix;
  
  for (int i = 0; i < num_matrices; i++) {
    NumericMatrix current_matrix = list_of_matrices[i];
    mat current_matrix_arma = as<mat>(current_matrix);
    combined_matrix = join_cols(combined_matrix, current_matrix_arma);
  }
  
  return trans(combined_matrix); // need transpose
}

arma::mat up_truncate_matrix(arma::mat x) {
  double threshold = 6.9;
  x.elem(find(x > threshold)).fill(threshold);
  return x;
}

arma::mat low_truncate_matrix(arma::mat x) {
  double threshold = -9.21;
  x.elem(find(x < threshold)).fill(threshold);
  return x;
}

// [[Rcpp::export]]
List get_opts(int L,
              Nullable<NumericVector> a_gamma = R_NilValue,
              Nullable<NumericVector> b_gamma = R_NilValue,
              Nullable<NumericVector> a_alpha = R_NilValue,
              Nullable<NumericVector> b_alpha = R_NilValue,
              Nullable<NumericVector> a_beta = R_NilValue,
              Nullable<NumericVector> b_beta = R_NilValue,
              Nullable<double> a = R_NilValue,
              Nullable<double> b = R_NilValue,
              Nullable<int> maxIter = R_NilValue,
              Nullable<int> thin = R_NilValue,
              Nullable<int> burnin = R_NilValue) {
  
  NumericVector a_gamma_vec = a_gamma.isNotNull() ? as<NumericVector>(a_gamma) : NumericVector(L, 1.0);
  NumericVector b_gamma_vec = b_gamma.isNotNull() ? as<NumericVector>(b_gamma) : NumericVector(L, 0.0);
  NumericVector a_alpha_vec = a_alpha.isNotNull() ? as<NumericVector>(a_alpha) : NumericVector(L, 0.0);
  NumericVector b_alpha_vec = b_alpha.isNotNull() ? as<NumericVector>(b_alpha) : NumericVector(L, 0.0);
  NumericVector a_beta_vec = a_beta.isNotNull() ? as<NumericVector>(a_beta) : NumericVector(L, 1.0);
  NumericVector b_beta_vec = b_beta.isNotNull() ? as<NumericVector>(b_beta) : NumericVector(L, 0.01);
  
  double a_val = a.isNotNull() ? Rcpp::as<double>(a) : 0.1;
  double b_val = b.isNotNull() ? Rcpp::as<double>(b) : 0.1;
  int maxIter_val = maxIter.isNotNull() ? Rcpp::as<int>(maxIter) : 4000;
  int thin_val = thin.isNotNull() ? Rcpp::as<int>(thin) : 10;
  int burnin_val = burnin.isNotNull() ? Rcpp::as<int>(burnin) : 1000;
  
  return List::create(Named("a_gamma") = a_gamma_vec,
                      Named("b_gamma") = b_gamma_vec,
                      Named("a_alpha") = a_alpha_vec,
                      Named("b_alpha") = b_alpha_vec,
                      Named("a_beta") = a_beta_vec,
                      Named("b_beta") = b_beta_vec,
                      Named("a") = a_val,
                      Named("b") = b_val,
                      Named("maxIter") = maxIter_val,
                      Named("thin") = thin_val,
                      Named("burnin") = burnin_val);
}

// [[Rcpp::depends(RcppArmadillo)]]
vec mean_ignore_nan_inf(const mat X) {
  vec col_mean = zeros<vec>(X.n_cols);
  for (unsigned int j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);
    vec finite_values = col(find_finite(col));
    if (finite_values.n_elem > 0) {
      col_mean(j) = mean(finite_values);
    } else {
      col_mean(j) = datum::nan; // set to NaN if all values are NaN or Inf
    }
  }
  return col_mean;
}
vec sd_ignore_nan_inf(const mat X) {
  vec col_sd = zeros<vec>(X.n_cols);
  for (unsigned int j = 0; j < X.n_cols; ++j) {
    vec col = X.col(j);
    vec finite_values = col(find_finite(col));
    if (finite_values.n_elem > 0) {
      col_sd(j) = stddev(finite_values);
    } else {
      col_sd(j) = datum::nan; // set to NaN if all values are NaN or Inf
    }
  }
  return col_sd;
}

List summarize_result(const List res) {
  List beta0res = res["beta0res"];
  List DeltaRes = res["DeltaRes"];
  List omegaRes = res["omegaRes"];
  
  int L = beta0res.size();
  int K = as<mat>(as<List>(as<List>(beta0res)[0])[0]).n_rows;
  
  // Initialize matrices to store results
  mat Estimate = mat(L, K, fill::zeros);
  mat Prob = mat(L, K, fill::zeros);
  mat Status = mat(L, K, fill::zeros);
  mat Pvalue = mat(L, K, fill::zeros);
  
  for (int s = 0; s < L; s++) {
    mat beta_est = do_call_rbind_vecs(as<List>(beta0res[s]));
    mat Delta_est = do_call_rbind_vecs(as<List>(DeltaRes[s]));
    mat omega_est = do_call_rbind_vecs(as<List>(omegaRes[s]));
    
    uvec all_zero = find(sum(Delta_est != 0, 0) <= 1);
    if (all_zero.n_elem > 0) {
      rowvec beta_est_mean = mean(beta_est.cols(all_zero), 0);
      rowvec beta_est_sd = stddev(beta_est.cols(all_zero), 0, 0) / sqrt(beta_est.n_rows);
      vec pvals = 2 * (1 - normcdf(abs(beta_est_mean / beta_est_sd)).t());
      for (uword i = 0; i < all_zero.n_elem; ++i) {
        Pvalue(s, all_zero(i)) = pvals(i);
        Estimate(s, all_zero(i)) = beta_est_mean[i];
      }
    }
    
    beta_est.elem(find(Delta_est == 0)).fill(datum::nan);
    uvec nonzero = find(sum(Delta_est != 0, 0) > 1);
    if (nonzero.n_elem > 0) {
      rowvec beta_est_mean = mean_ignore_nan_inf(beta_est.cols(nonzero)).t();
      rowvec beta_est_sd = sd_ignore_nan_inf(beta_est.cols(nonzero)).t();
      vec pvals = 2 * (1 - normcdf(abs(beta_est_mean / beta_est_sd)).t());
      for (uword i = 0; i < nonzero.n_elem; ++i) {
        Pvalue(s, nonzero(i)) = pvals(i);
        Estimate(s, nonzero(i)) = beta_est_mean[i];
      }
    }
    
    // Prob.row(s) = mean(omega_est, 0);
    // Status.row(s) = mean(Delta_est, 0);
  }
  return List::create(Named("Estimate") = Estimate,
                      Named("Pvalue") = Pvalue);
}

rowvec check_missing_in_matrix(const mat x) {
  rowvec result(x.n_cols);
  for (unsigned int j = 0; j < x.n_cols; ++j) {
    // Use find_finite to get indices of all finite values in the column
    uvec finite_indices = find_finite(x.col(j));
    // If find_finite returns an empty vector, the column is entirely non-finite
    result(j) = finite_indices.empty() ? 1 : 0;
  }
  return result;
}

// [[Rcpp::export]]
arma::mat check_missing(List gammah) {
  // Determine the number of matrices in the list and the number of columns in the first matrix
  int n_matrices = gammah.size();
  mat first_matrix = as<mat>(gammah[0]);
  int n_cols = first_matrix.n_cols;
  
  // Create a matrix to store the results
  mat result = zeros<mat>(n_matrices, n_cols);
  
  // Iterate over each matrix in the list
  for(int i = 0; i < n_matrices; ++i) {
    mat current_matrix = as<mat>(gammah[i]);
    result.row(i) = check_missing_in_matrix(current_matrix);
  }
  
  return result;
}

// Function to generate unit vector
colvec ei(int i, int n) {
  colvec e = zeros<colvec>(n);
  e(i - 1) = 1; // Adjust for C++ 0-based indexing
  return e;
}

// [[Rcpp::export]]
arma::mat generate_transformation_indicator(const arma::mat gammah_elem) {
  // Vector to hold indices of columns that are not entirely NA
  uvec not_all_na_cols_indices;

  // Iterate over each column to check for at least one finite value using find_finite
  for (unsigned int j = 0; j < gammah_elem.n_cols; ++j) {
    uvec finite_indices = find_finite(gammah_elem.col(j));
    if (finite_indices.n_elem > 0) {
      // If there are finite values, the column is not all NA
      not_all_na_cols_indices.insert_rows(not_all_na_cols_indices.size(), uvec({j}));
    }
  }

  // Create an indicator matrix with the appropriate size
  mat indicator = zeros<mat>(gammah_elem.n_cols, not_all_na_cols_indices.size());

  // Fill the indicator matrix for each non-NA column
  for (size_t i = 0; i < not_all_na_cols_indices.size(); ++i) {
    indicator(not_all_na_cols_indices(i), i) = 1;
  }

  // Assign NA to rows that are all zeros
  for (unsigned int i = 0; i < indicator.n_rows; ++i) {
    if (accu(indicator.row(i)) == 0) {
      indicator.row(i).fill(datum::nan); // Fill the row with NA values
    }
  }

  return indicator;
}

// [[Rcpp::export]]
arma::mat keep_nonmissing_column(const arma::mat X) {
  // Initialize a vector to store indices of non-missing columns
  uvec non_missing_cols;

  for (unsigned int j = 0; j < X.n_cols; ++j) {
    // Check if the column has any missing (NaN) values
    if (!X.col(j).has_nan()) {
      // If no missing values are found, store the column index
      non_missing_cols.insert_rows(non_missing_cols.size(), uvec({j}));
    }
  }

  // Use the indices to select non-missing columns
  return X.cols(non_missing_cols);
}

// [[Rcpp::export]]
List spca(const arma::mat x, const arma::mat y) {
  
  int n = x.n_rows;
  arma::mat H = arma::eye<mat>(n, n) - arma::ones<mat>(n, n) / n;
  arma::mat L;
  
  if (y.n_elem == 0) { // Equivalent to checking if y is a diagonal matrix in R version
    L = arma::eye<mat>(n, n);
  } else {
    L = y; // Directly using y if it's a matrix
  }
  
  // Calculation of Q matrix
  arma::mat Q = x.t() * H * L * H * x;
  
  // Eigen decomposition
  List evd = Rcpp::as<List>(eigen(Q, true));
  mat eigenvectors = evd["vectors"];
  
  // Optional return of transformed x
  arma::mat x_transformed;
  x_transformed = x * eigenvectors;
  
  // Setting column names for eigenvectors (not directly supported in Armadillo, handled at R level)
  
  return List::create(Named("vectors") = eigenvectors,
                      Named("Q") = Q,
                      Named("x") = x_transformed);
}

// [[Rcpp::export]]
List VarCompTest_cpp(const arma::vec y, const arma::mat Z) {
  // Calculate residuals as y - mean(y)
  vec resid = y - mean(y);
  double s2 = dot(resid, resid) / (resid.n_elem - 1);
  
  // Compute Q.Temp as t(resid) %*% Z
  mat Q_Temp = resid.t() * Z;
  
  // Compute Q
  mat Q = Q_Temp * Q_Temp.t() / s2 / 2;
  
  // Construct X1 and calculate W.1
  mat X1 = ones<vec>(Z.n_rows);
  mat W_1 = Z.t() * Z - (Z.t() * X1) * inv(X1.t() * X1) * (X1.t() * Z);
  
  // // Placeholder for calling SKAT::Get_Davies_PVal or equivalent
  // List out = Get_Davies_PVal(Q,W_1); // Needs replacement with the actual call or computation
  
  return List::create(Named("Q") = Q,
                      Named("W") = W_1);
}

// // [[Rcpp::export]]
// arma::mat cppMIDAS(arma::mat gammah){
//   List X_conv = convert(dataframe(gammah));
//   List complete_data = complete(train(X_conv,Named("training_epochs")=20,Named("seed")=1),1);
//   mat complete_data_mat = as<mat>(datamatrix(complete_data[0]));
//   return(complete_data_mat);
// }

// [[Rcpp::export]]
arma::mat cppMIDAS(arma::mat gammah){
  mat complete_data_mat = as<mat>(MIDAS(gammah));
  return(complete_data_mat);
}

// [[Rcpp::export]]
arma::mat cppmissForest(arma::mat gammah){
  List res = missForest(gammah);
  mat complete_data_mat = as<mat>(res["ximp"]);
  return(complete_data_mat);
}


List mintMR_multi_omics(const List gammah, const List Gammah,
                        const List se1, const List se2,
                        const List corr_mat, const List group, const List opts,
                        const arma::mat Lambda,
                        bool display_progress=true,
                        int CC = 2, int PC1 = 1, int PC2 = 1) {
  int L = gammah.length();
  int K = as<mat>(gammah[0]).n_cols;
  IntegerVector p(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
  }
  mat Lambda_Offdiag = Lambda;
  Lambda_Offdiag.diag().zeros();
  
  List corr_mat_Offdiag(L);
  for (int i = 0; i < L; i++){
    mat matrix = as<mat>(corr_mat[i]);
    matrix.diag().zeros();
    corr_mat_Offdiag[i] = matrix;
  }
  
  
  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];
  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K, fill::ones) * aval;
    b[i] = vec(K, fill::ones) * bval;
  }
  double u0 = 0.1;
  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;
  
  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;
  vec sgal2xi2 = sgal2 % xi2;
  
  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K, fill::ones) * 0.1;
    omega[i] = vec(K, fill::ones) * 0.1;
  }
  
  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K, fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K, fill::ones) * 0.01;
    
    sgga2[i] = vec(p[i], fill::ones) * 0.01;
    m0save[i] = mat(p[i], K, fill::ones) * 0.01;
    m1save[i] = mat(p[i], K, fill::ones) * 0.01;
  }
  
  
  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);
    
    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K, fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K, fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K, fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K, fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }
  
  
  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }
  
  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);
  
  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);
    
    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);
    
    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);
    
    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);
    
    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);
    
    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);
    
    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }
  
  
  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;
  Progress pgbar((maxIter + burnin), display_progress);
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }
    
    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;
      
      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + Lambda(0,0) * as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = (Lambda(0,0) * diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      for (int k1 = 0; k1 < K; k1++) {
        mut1 = mut1 + Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(gammah[ell]).col(k1);
        mut1 = mut1 - Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mu[ell]).col(k1);
      }
      mut1 = v0t * mut1;
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        mat v1t;
        vec mut1;
        v1t = inv(invsgga2[k] * as<mat>(I[ell]) + Lambda(k+1,k+1) * as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
        mut1 = as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) +
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(Gammah[ell])-
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mut[ell]);
        for (int k1 = 0; k1 < K; k1++) {
          mut1 = mut1 + Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(gammah[ell]).col(k1);
          if(k1!=k){
            mut1 = mut1 - Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(mu[ell]).col(k1);
          }
        }
        mut1 = v1t * mut1;
        mat mu_ell = as<mat>(mu[ell]);
        mu_ell.col(k) = mvnrnd(mut1, v1t);
        mu[ell] = mu_ell;
      }
      
      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        
        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);
        
        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);
        
        Delta[ell] = Delta_ell;
      }
      
      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;
      
      
      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }
      
      
      beta0[ell] = beta0_ell;
      
      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);
      
      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));
      
      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];
      
      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      double ta_gamma;
      double tb_gamma;
      int K1 = as<uvec>(group[0]).n_elem;
      ta_gamma = a_gamma[ell] + K1 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[0]) - 1))/2;
      double sgga2_grp1 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      int K2 = as<uvec>(group[1]).n_elem;
      ta_gamma = a_gamma[ell] + K2 * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1)%as<mat>(mu[ell]).cols(as<uvec>(group[1]) - 1))/2;
      double sgga2_grp2 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K1; k++) {
        sgga2_ell[k] = sgga2_grp1;
      }
      for (int k = 0; k < K2; k++) {
        sgga2_ell[k+K1] = sgga2_grp2;
      }
      sgga2[ell] = sgga2_ell;
      
      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/(ta_beta - 1);
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);
        
        mat omega_ell = as<mat>(omega[ell]);
        
        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }
      
      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = beta0[ell];
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = omega[ell];
          as<List>(DeltaRes[ell])[l] = Delta[ell];
        }
      }
    }
    
    mat alpha_all = do_call_rbind_vecs(omega);
    mat colmean_alpha = mean(alpha_all, 0);
    
    
    mat alpha_all1 = alpha_all.cols(as<uvec>(group[0]) - 1);
    mat alpha_all2 = alpha_all.cols(as<uvec>(group[1]) - 1);
    
    mat U1 = log(alpha_all1 / (1 - alpha_all1)) - u0;
    mat U2 = log(alpha_all2 / (1 - alpha_all2)) - u0;
    
    U1 = up_truncate_matrix(U1);
    U2 = up_truncate_matrix(U2);
    U1 = low_truncate_matrix(U1);
    U2 = low_truncate_matrix(U2);
    
    mat norm_U1c = normalize_mat(U1);
    mat norm_U2c = normalize_mat(U2);
    mat colmean1c = mean(U1, 0);
    mat colmean2c = mean(U2, 0);
    mat sd1c = stddev(U1, 0, 0);
    mat sd2c = stddev(U2, 0, 0);
    
    List ccas = Rcpp::as<List>(cc(U1, U2));
    mat Ahat = ccas["xcoef"];
    mat Bhat = ccas["ycoef"];
    mat XX = U1 * Ahat;
    mat YY = U2 * Bhat;
    mat X_est1 = XX.cols(0, CC - 1) * pinv(Ahat.cols(0, CC - 1));
    mat X_est2 = YY.cols(0, CC - 1) * pinv(Bhat.cols(0, CC - 1));
    
    mat X_res1 = U1 - X_est1;
    mat X_res2 = U2 - X_est2;
    
    // perform PCA
    mat U;
    vec s;
    mat V;
    mat norm_U1 = normalize_mat(X_res1);
    mat norm_U2 = normalize_mat(X_res2);
    
    mat colmean1 = mean(X_res1, 0);
    mat colmean2 = mean(X_res2, 0);
    mat sd1 = stddev(X_res1, 0, 0);
    mat sd2 = stddev(X_res2, 0, 0);
    
    svd(U, s, V, norm_U1);
    mat X_red1 = U.cols(0, PC1 - 1) * diagmat(s.subvec(0, PC1 - 1)) * trans(V.cols(0, PC1 - 1));
    
    svd(U, s, V, normalize_mat(X_res2));
    mat X_red2 = U.cols(0, PC2 - 1) * diagmat(s.subvec(0, PC2 - 1)) * trans(V.cols(0, PC2 - 1));
    
    X_red1 = X_red1 * diagmat(sd1);
    for (int j = 0; j < X_red1.n_cols; j++) {
      X_red1.col(j) += colmean1[j];
    }
    X_red2 = X_red2 * diagmat(sd2);
    for (int j = 0; j < X_red2.n_cols; j++) {
      X_red2.col(j) += colmean2[j];
    }
    
    mat X_est = join_rows(X_est1, X_est2);
    mat X_red = join_rows(X_red1, X_red2);
    
    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      current_omega = 1 / (1 + exp(- X_red.row(ell) - X_est.row(ell) - u0));
      // current_omega = 1 / (1 + exp(-X_red.row(ell) - u0));
      omega[ell] = trans(current_omega);
    }
    
    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }
  
  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes
  );
  return res;
}
List mintMR_single_omics(const List gammah, const List Gammah,
                         const List se1, const List se2,
                         const List corr_mat, const List opts,
                         const arma::mat Lambda,
                         bool display_progress=true,
                         int PC1 = 1) {
  int L = gammah.length();
  int K = as<mat>(gammah[0]).n_cols;
  IntegerVector p(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
  }
  mat Lambda_Offdiag = Lambda;
  Lambda_Offdiag.diag().zeros();
  
  List corr_mat_Offdiag(L);
  for (int i = 0; i < L; i++){
    mat matrix = as<mat>(corr_mat[i]);
    matrix.diag().zeros();
    corr_mat_Offdiag[i] = matrix;
  }
  
  
  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];
  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K, fill::ones) * aval;
    b[i] = vec(K, fill::ones) * bval;
  }
  double u0 = 0.1;
  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;
  
  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;
  vec sgal2xi2 = sgal2 % xi2;
  
  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K, fill::ones) * 0.1;
    omega[i] = vec(K, fill::ones) * 0.1;
  }
  
  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K, fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K, fill::ones) * 0.01;
    
    sgga2[i] = vec(p[i], fill::ones) * 0.01;
    m0save[i] = mat(p[i], K, fill::ones) * 0.01;
    m1save[i] = mat(p[i], K, fill::ones) * 0.01;
  }
  
  
  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);
    
    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K, fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K, fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K, fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K, fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }
  
  
  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }
  
  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);
  
  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);
    
    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);
    
    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);
    
    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);
    
    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);
    
    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);
    
    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }
  
  
  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;
  Progress pgbar((maxIter + burnin), display_progress);
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }
    
    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;
      
      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + Lambda(0,0) * as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = (Lambda(0,0) * diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      for (int k1 = 0; k1 < K; k1++) {
        mut1 = mut1 + Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(gammah[ell]).col(k1);
        mut1 = mut1 - Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mu[ell]).col(k1);
      }
      mut1 = v0t * mut1;
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        mat v1t;
        vec mut1;
        v1t = inv(invsgga2[k] * as<mat>(I[ell]) + Lambda(k+1,k+1) * as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
        mut1 = as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) +
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(Gammah[ell])-
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mut[ell]);
        for (int k1 = 0; k1 < K; k1++) {
          mut1 = mut1 + Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(gammah[ell]).col(k1);
          if(k1!=k){
            mut1 = mut1 - Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(mu[ell]).col(k1);
          }
        }
        mut1 = v1t * mut1;
        mat mu_ell = as<mat>(mu[ell]);
        mu_ell.col(k) = mvnrnd(mut1, v1t);
        mu[ell] = mu_ell;
      }
      
      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        
        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);
        
        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);
        
        Delta[ell] = Delta_ell;
      }
      
      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;
      
      
      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }
      
      
      beta0[ell] = beta0_ell;
      
      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);
      
      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));
      
      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];
      
      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      double ta_gamma;
      double tb_gamma;
      ta_gamma = a_gamma[ell] + K * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell])%as<mat>(mu[ell]))/2;
      double sgga2_tmp = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K; k++) {
        sgga2_ell[k] = sgga2_tmp;
      }
      sgga2[ell] = sgga2_ell;
      
      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/(ta_beta - 1);
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);
        
        mat omega_ell = as<mat>(omega[ell]);
        
        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }
      
      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = beta0[ell];
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = omega[ell];
          as<List>(DeltaRes[ell])[l] = Delta[ell];
        }
      }
    }
    
    mat alpha_all = do_call_rbind_vecs(omega);
    mat colmean_alpha = mean(alpha_all, 0);
    
    mat U1 = log(alpha_all / (1 - alpha_all)) - u0;
    
    U1 = up_truncate_matrix(U1);
    U1 = low_truncate_matrix(U1);
    
    // perform PCA
    mat U;
    vec s;
    mat V;
    mat norm_U1 = normalize_mat(U1);
    
    mat colmean1 = mean(U1, 0);
    mat sd1 = stddev(U1, 0, 0);
    
    svd(U, s, V, norm_U1);
    mat X_red1 = U.cols(0, PC1 - 1) * diagmat(s.subvec(0, PC1 - 1)) * trans(V.cols(0, PC1 - 1));
    
    X_red1 = X_red1 * diagmat(sd1);
    for (int j = 0; j < X_red1.n_cols; j++) {
      X_red1.col(j) += colmean1[j];
    }
    
    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      current_omega = 1 / (1 + exp(-X_red1.row(ell) - u0));
      omega[ell] = trans(current_omega);
    }
    
    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }
  
  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes
  );
  return res;
}
List mintMR_single_omics_supervised(const List gammah, const List Gammah,
                                    const List se1, const List se2,
                                    const mat reference,
                                    const List corr_mat, const List opts,
                                    const arma::mat Lambda,
                                    bool display_progress=true,
                                    int PC1 = 1) {
  int L = gammah.length();
  int K = as<mat>(gammah[0]).n_cols;
  IntegerVector p(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
  }
  mat Lambda_Offdiag = Lambda;
  Lambda_Offdiag.diag().zeros();
  
  List corr_mat_Offdiag(L);
  for (int i = 0; i < L; i++){
    mat matrix = as<mat>(corr_mat[i]);
    matrix.diag().zeros();
    corr_mat_Offdiag[i] = matrix;
  }
  
  
  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];
  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K, fill::ones) * aval;
    b[i] = vec(K, fill::ones) * bval;
  }
  double u0 = 0.1;
  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;
  
  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;
  vec sgal2xi2 = sgal2 % xi2;
  
  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K, fill::ones) * 0.1;
    omega[i] = vec(K, fill::ones) * 0.1;
  }
  
  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L), QRes(L), WRes(L), PRes(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K, fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K, fill::ones) * 0.01;
    
    sgga2[i] = vec(p[i], fill::ones) * 0.01;
    m0save[i] = mat(p[i], K, fill::ones) * 0.01;
    m1save[i] = mat(p[i], K, fill::ones) * 0.01;
  }
  
  
  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);
    QRes[ell] = List(numsave);
    WRes[ell] = List(numsave);
    PRes[ell] = List(numsave);
    
    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K, fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K, fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K, fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K, fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }
  
  
  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }
  
  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);
  
  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);
    
    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);
    
    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);
    
    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);
    
    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);
    
    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);
    
    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }
  
  
  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;
  Progress pgbar((maxIter + burnin), display_progress);
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }
    
    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;
      
      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + Lambda(0,0) * as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = (Lambda(0,0) * diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      for (int k1 = 0; k1 < K; k1++) {
        mut1 = mut1 + Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(gammah[ell]).col(k1);
        mut1 = mut1 - Lambda(0,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mu[ell]).col(k1);
      }
      mut1 = v0t * mut1;
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        mat v1t;
        vec mut1;
        v1t = inv(invsgga2[k] * as<mat>(I[ell]) + Lambda(k+1,k+1) * as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
        mut1 = as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) +
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(Gammah[ell])-
          Lambda(0,k+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mut[ell]);
        for (int k1 = 0; k1 < K; k1++) {
          mut1 = mut1 + Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(gammah[ell]).col(k1);
          if(k1!=k){
            mut1 = mut1 - Lambda(k+1,k1+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(mu[ell]).col(k1);
          }
        }
        mut1 = v1t * mut1;
        mat mu_ell = as<mat>(mu[ell]);
        mu_ell.col(k) = mvnrnd(mut1, v1t);
        mu[ell] = mu_ell;
      }
      
      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        
        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);
        
        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);
        
        Delta[ell] = Delta_ell;
      }
      
      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;
      
      // ----------------------- //
      // VarCompTest Test
      // ----------------------- //
      mat mu_delta = as<mat>(mu[ell]);//*diagmat(as<mat>(Delta[ell]));
      List VarCompTest_out = VarCompTest_cpp(Gammah[ell],mu_delta);
      
      
      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }
      
      
      beta0[ell] = beta0_ell;
      
      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);
      
      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));
      
      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];
      
      if(sgal2xi2[ell] < 1e-7){
        sgal2xi2[ell] = 1e-7;
      }
      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      double ta_gamma;
      double tb_gamma;
      ta_gamma = a_gamma[ell] + K * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell])%as<mat>(mu[ell]))/2;
      double sgga2_tmp = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K; k++) {
        sgga2_ell[k] = sgga2_tmp;
      }
      sgga2[ell] = sgga2_ell;
      
      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/(ta_beta - 1);
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);
        
        mat omega_ell = as<mat>(omega[ell]);
        
        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }
      
      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = beta0[ell];
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = omega[ell];
          as<List>(DeltaRes[ell])[l] = Delta[ell];
          as<List>(mutRes[ell])[l] = mut[ell];
          as<List>(muRes[ell])[l] = mu[ell];
          as<List>(QRes[ell])[l] = VarCompTest_out["Q"];
          as<List>(WRes[ell])[l] = VarCompTest_out["W"];
        }
      }
    }
    
    mat alpha_all = do_call_rbind_vecs(omega);
    mat colmean_alpha = mean(alpha_all, 0);
    
    mat U1 = log(alpha_all / (1 - alpha_all)) - u0;
    
    U1 = up_truncate_matrix(U1);
    U1 = low_truncate_matrix(U1);
    
    // perform supervised PCA
    mat norm_U1 = normalize_mat(U1);
    
    mat colmean1 = mean(U1, 0);
    mat sd1 = stddev(U1, 0, 0);
    
    List spca_result;
    spca_result = spca(norm_U1, reference*trans(reference));
    
    mat X_red1 = as<mat>(spca_result["x"]).cols(0, PC1 - 1) *
      trans(as<mat>(spca_result["vectors"]).cols(0, PC1 - 1));
    
    X_red1 = X_red1 * diagmat(sd1);
    for (int j = 0; j < X_red1.n_cols; j++) {
      X_red1.col(j) += colmean1[j];
    }
    
    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      current_omega = 1 / (1 + exp(-X_red1.row(ell) - u0));
      omega[ell] = trans(current_omega);
    }
    
    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }
  
  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes,
    _["QRes"] = QRes,
    _["WRes"] = WRes,
    _["PRes"] = PRes
  );
  return res;
}
// [[Rcpp::export]]
List mintMR_Impute_MVL(List gammah, const List Gammah,
                       List se1, const List se2,
                       const List corr_mat, const List group, const List opts,
                       const arma::mat Lambda,
                       bool display_progress=true,
                       int latent_dim = 2,
                       int CC = 2, int PC1 = 1, int PC2 = 1,
                       String missing_method = "missForest", // Added parameter for missing value handling
                       String mvl_method = "DVCCA",
                       int epochs = 5,
                       bool fast_impute = true) {
  cout << "Now running mintMR_Impute_MVL ..." << endl;

  int L = gammah.length();
  IntegerVector p(L);
  IntegerVector K(L);
  for (int i = 0; i < L; i++) {
    p[i] = as<mat>(gammah[i]).n_rows;
    K[i] = keep_nonmissing_column(as<mat>(gammah[i])).n_cols;
  }
  mat Lambda_Offdiag = Lambda;
  Lambda_Offdiag.diag().zeros();

  List corr_mat_Offdiag(L);
  for (int i = 0; i < L; i++){
    mat matrix = as<mat>(corr_mat[i]);
    matrix.diag().zeros();
    corr_mat_Offdiag[i] = matrix;
  }
  
  // ---------------------------------------- //
  // added allowing missingness
  // ---------------------------------------- //
  List transformation_indicator(L);
  for (int i = 0; i < L; i++) {
    mat indicator = generate_transformation_indicator(as<mat>(gammah[i]));
    transformation_indicator[i] = indicator;
  }
  
  for (int i = 0; i < L; i++) {
    mat gammah_i = keep_nonmissing_column(as<mat>(gammah[i]));
    mat se1_i = keep_nonmissing_column(as<mat>(se1[i]));
    gammah[i] = wrap(gammah_i);
    se1[i] = wrap(se1_i);
  }
  
  // ---------------------------------------- //

  vec a_gamma = as<vec>(opts["a_gamma"]);
  vec b_gamma = as<vec>(opts["b_gamma"]);
  vec a_alpha = as<vec>(opts["a_alpha"]);
  vec b_alpha = as<vec>(opts["b_alpha"]);
  vec a_beta = as<vec>(opts["a_beta"]);
  vec b_beta = as<vec>(opts["b_beta"]);
  double aval = opts["a"];
  double bval = opts["b"];
  List a(L), b(L);
  for (int i = 0; i < L; i++) {
    a[i] = vec(K[i], fill::ones) * aval;
    b[i] = vec(K[i], fill::ones) * bval;
  }
  double u0 = 0.1;
  int maxIter = as<int>(opts["maxIter"]);
  int thin = as<int>(opts["thin"]);
  int burnin = as<int>(opts["burnin"]) - 1;
  vec sgal2 = vec(L, fill::ones) * 0.01;
  vec sgbeta2 = vec(L, fill::ones) * 0.01;
  vec xi2 = vec(L, fill::ones) * 0.01;
  vec sgal2xi2 = sgal2 % xi2;
  List beta0(L), omega(L);
  for (int i = 0; i < L; i++) {
    beta0[i] = vec(K[i], fill::ones) * 0.1;
    omega[i] = vec(K[i], fill::ones) * 0.1;
  }
  int numsave = maxIter / thin + 1;
  List Sgga2Res(L), Sgal2Res(L), Sgbeta2Res(L), Delta(L), beta0res(L), DeltaRes(L), omegaRes(L), mut(L), mu(L), mutRes(L), muRes(L), sgga2(L), QRes(L), WRes(L), PRes(L);
  List m0save(L), m1save(L);
  for (int i = 0; i < L; i++) {
    Delta[i] = vec(K[i], fill::zeros);
    mut[i] = vec(p[i], fill::ones) * 0.01;
    mu[i] = mat(p[i], K[i], fill::ones) * 0.01;
    
    sgga2[i] = vec(p[i], fill::ones) * 0.01;
    m0save[i] = mat(p[i], K[i], fill::ones) * 0.01;
    m1save[i] = mat(p[i], K[i], fill::ones) * 0.01;
  }

  
  for (int ell = 0; ell < L; ell++) {
    beta0res[ell] = List(numsave);
    omegaRes[ell] = List(numsave);
    DeltaRes[ell] = List(numsave);
    Sgga2Res[ell] = List(numsave);
    Sgal2Res[ell] = List(numsave);
    Sgbeta2Res[ell] = List(numsave);
    mutRes[ell] = List(numsave);
    muRes[ell] = List(numsave);
    QRes[ell] = List(numsave);
    WRes[ell] = List(numsave);
    PRes[ell] = List(numsave);
    
    for (int l = 0; l < numsave; l++) {
      as<List>(beta0res[ell])[l] = vec(K[ell], fill::ones);
      as<List>(omegaRes[ell])[l] = vec(K[ell], fill::ones);
      as<List>(DeltaRes[ell])[l] = vec(K[ell], fill::ones);
      as<List>(Sgga2Res[ell])[l] = vec(K[ell], fill::ones);
      as<List>(Sgal2Res[ell])[l] = 1;
      as<List>(Sgbeta2Res[ell])[l] = 1;
    }
  }
  

  List sG2(L), sg2(L), invsG2(L), invsg2(L);
  for (int i = 0; i < L; i++) {
    sG2[i] = as<mat>(se2[i]) % as<mat>(se2[i]);
    sg2[i] = as<mat>(se1[i]) % as<mat>(se1[i]);
    invsG2[i] = 1 / as<mat>(sG2[i]);
    invsg2[i] = 1 / as<mat>(sg2[i]);
  }
  
  List S_GRS_Ginv(L), S_gRS_ginv(L), S_GinvRS_Ginv(L), S_ginvRS_ginv(L), S_GRS_G(L), S_gRS_g(L);
  
  for (int i = 0; i < L; i++) {
    vec se2_i = se2[i];
    mat se1_i = se1[i];
    mat corr_mat_i = as<mat>(corr_mat[i]);
    
    // calculate S_GRS_Ginv
    mat S_GRS_Ginv_i = diagmat(se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GRS_Ginv[i] = wrap(S_GRS_Ginv_i);
    
    // calculate S_gRS_ginv
    List S_gRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_ginv_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_gRS_ginv[i] = wrap(S_gRS_ginv_i);
    
    // calculate S_GinvRS_Ginv
    mat S_GinvRS_Ginv_i = diagmat(1/se2_i) * corr_mat_i * diagmat(1/se2_i);
    S_GinvRS_Ginv[i] = wrap(S_GinvRS_Ginv_i);
    
    // calculate S_ginvRS_ginv
    List S_ginvRS_ginv_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_ginvRS_ginv_i[j] = diagmat(1/se1_i.col(j)) * corr_mat_i * diagmat(1/se1_i.col(j));
    }
    S_ginvRS_ginv[i] = wrap(S_ginvRS_ginv_i);
    
    // calculate S_GRS_G
    mat S_GRS_G_i = diagmat(se2_i) * corr_mat_i * diagmat(se2_i);
    S_GRS_G[i] = wrap(S_GRS_G_i);
    
    // calculate S_gRS_g
    List S_gRS_g_i(se1_i.n_cols);
    for (int j = 0; j < se1_i.n_cols; j++) {
      S_gRS_g_i[j] = diagmat(se1_i.col(j)) * corr_mat_i * diagmat(se1_i.col(j));
    }
    S_gRS_g[i] = wrap(S_gRS_g_i);
  }
  
  
  List I(L);
  for (int i = 0; i < L; i++) {
    int ptmp = p[i];
    mat I_temp = eye<mat>(ptmp, ptmp);
    I[i] = I_temp;
  }
  int l = 0;
  int ell, iter;
  Progress pgbar((maxIter + burnin), display_progress);
  mat U_complete;
  for (iter = 1; iter <= (maxIter + burnin); iter++) {
    pgbar.increment();
    if (Progress::check_abort()) {
      List res = List::create(
        _["beta0res"] = beta0res,
        _["Sgga2Res"] = Sgga2Res,
        _["Sgal2Res"] = Sgal2Res,
        _["omegaRes"] = omegaRes,
        _["DeltaRes"] = DeltaRes
      );
      return res;
    }
    
    for (ell = 0; ell < L; ell++) {
      vec invsgga2 = 1. / as<vec>(sgga2[ell]);
      vec invsgal2xi2 = 1. / sgal2xi2;

      // ----------------------- //
      // Parameters for Gamma
      // ----------------------- //
      mat v0t = inv(invsgal2xi2[ell] * as<mat>(I[ell]) + Lambda(0,0) * as<mat>(S_GinvRS_Ginv[ell]));
      mat mut1 = (Lambda(0,0) * diagmat(1 / as<vec>(sG2[ell])) * as<mat>(Gammah[ell]) +
        invsgal2xi2[ell] * (as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell]))));
      for (int k1 = 0; k1 < K[ell]; k1++) {
        uvec lam_k1 = find(as<mat>(transformation_indicator[ell]).col(k1)==1);
        mut1 = mut1 + Lambda(0,lam_k1[0]+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(gammah[ell]).col(k1);
        mut1 = mut1 - Lambda(0,lam_k1[0]+1) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mu[ell]).col(k1);
      }
      mut1 = v0t * mut1;
      mat mut_ell = mvnrnd(mut1, v0t);
      mut[ell] = mut_ell;
      // ----------------------- //
      // Parameters for gamma
      // ----------------------- //

      for (int k = 0; k < K[ell]; k++) {
        uvec lam_k = find(as<mat>(transformation_indicator[ell]).col(k)==1);
        
        mat v1t;
        vec mut1;
        v1t = inv(invsgga2[k] * as<mat>(I[ell]) + Lambda(lam_k[0]+1,lam_k[0]+1) * as<mat>(as<List>(S_ginvRS_ginv[ell])[k]) + as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * as<mat>(I[ell]));
        mut1 = as<mat>(Delta[ell])[k] * as<mat>(beta0[ell])[k] * invsgal2xi2[ell] * (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * (as<mat>(Delta[ell]) % as<mat>(beta0[ell])), 1) + as<mat>(beta0[ell])[k] * as<mat>(mu[ell]).col(k)) +
          Lambda(0,lam_k[0]+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(Gammah[ell])-
          Lambda(0,lam_k[0]+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * diagmat(1 / as<mat>(se2[ell])) * as<mat>(mut[ell]);
        for (int k1 = 0; k1 < K[ell]; k1++) {
          uvec lam_k1 = find(as<mat>(transformation_indicator[ell]).col(k1)==1);
          mut1 = mut1 + Lambda(lam_k[0]+1,lam_k1[0]+1) * diagmat(1 / as<mat>(se1[ell]).col(k)) * diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(gammah[ell]).col(k1);
          if(k1!=k){
            mut1 = mut1 - Lambda(lam_k[0]+1,lam_k1[0]+1) * 
                                   diagmat(1 / as<mat>(se1[ell]).col(k)) * as<mat>(corr_mat[ell]) * 
                                   diagmat(1 / as<mat>(se1[ell]).col(k1)) * as<mat>(mu[ell]).col(k1);
          }
        }
        
        mut1 = v1t * mut1;
        mat mu_ell = as<mat>(mu[ell]);
        mu_ell.col(k) = mvnrnd(mut1, v1t);
        mu[ell] = mu_ell;
      }

      // ----------------------- //
      // Update Delta;
      // ----------------------- //
      double pr0, pr1, prob;
      mat m0_ell = as<mat>(mu[ell]);
      mat m1_ell = as<mat>(mu[ell]);
      for (int k = 0; k < K[ell]; k++) {
        vec m0 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k) + as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        vec m1 = as<mat>(mu[ell])*(as<mat>(Delta[ell]) % as<mat>(beta0[ell])) - as<mat>(Delta[ell])[k]*as<mat>(beta0[ell])[k]*as<mat>(mu[ell]).col(k);
        
        pr0 = as<mat>(omega[ell])[k];
        pr1 = (1 - as<mat>(omega[ell])[k]) *
          exp(-0.5 * (accu((as<mat>(mut[ell]) - m1)%(as<mat>(mut[ell]) - m1)) -
          accu((as<mat>(mut[ell]) - m0)%(as<mat>(mut[ell]) - m0))) / sgal2xi2[ell]);
        
        prob = pr0 / (pr0 + pr1);
        vec Delta_ell = as<vec>(Delta[ell]);
        Delta_ell[k] = R::rbinom(1, prob);
        
        Delta[ell] = Delta_ell;
      }
      
      m0save[ell] = m0_ell;
      m1save[ell] = m1_ell;
      // ----------------------- //
      // VarCompTest Test
      // ----------------------- //
      mat mu_delta = as<mat>(mu[ell]);
      List VarCompTest_out = VarCompTest_cpp(Gammah[ell],mu_delta);
      
      // ----------------------- //
      // Update beta0, beta1;
      // ----------------------- //
      mat beta0_ell = as<mat>(beta0[ell]);
      for (int k = 0; k < K[ell]; k++) {
        if (as<mat>(Delta[ell])[k] == 1) {
          double sig2b0t = 1 / (accu(as<mat>(mu[ell]).col(k) % as<mat>(mu[ell]).col(k)) / sgal2xi2[ell] + 1 / sgbeta2[ell]);
          double mub0 = accu(as<mat>(mu[ell]).col(k) %
                             (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % beta0_ell),1) + as<mat>(Delta[ell])[k] * beta0_ell[k] * as<mat>(mu[ell]).col(k))) *
                             invsgal2xi2[ell] * sig2b0t;
          beta0_ell[k] = mub0 + randn() * sqrt(sig2b0t);
        } else {
          double sig2b0t = sgbeta2[ell];
          beta0_ell[k] = randn() * sqrt(sig2b0t);
        }
      }
      
      // cout << "beta0_ell -- " << beta0_ell << endl;
      beta0[ell] = beta0_ell;
      // ----------------------- //
      // Update sigma_alpha;
      // ----------------------- //
      double err0 = accu((as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)) % (as<mat>(mut[ell]) - sum(as<mat>(mu[ell]) * diagmat(as<mat>(Delta[ell]) % as<mat>(beta0[ell])),1)));
      double ta_alpha = a_alpha[ell] + p[ell] / 2;
      double tb_alpha = b_alpha[ell] + err0 / (2 * xi2[ell]);
      // cout << "ta_alpha -- " << ta_alpha << endl;
      // cout << "tb_alpha -- " << ta_alpha << endl;
      sgal2[ell] = 1 / randg<double>(distr_param(ta_alpha,1/tb_alpha));
      // ----------------------- //
      // Update xi2
      // ----------------------- //
      double taxi2 = p[ell] / 2;
      double tbxi2 = 0.5*accu(err0)/sgal2[ell];
      xi2[ell] =  1 / randg<double>(distr_param(taxi2, 1/tbxi2));
      sgal2xi2[ell] = sgal2[ell]*xi2[ell];
      // ----------------------- //
      // Update sgga2
      // ----------------------- //
      mat indicator_group, indicator_group_NA_remove;
      uvec group0, group1;
      indicator_group = as<mat>(transformation_indicator[ell]).rows(as<uvec>(group[0]) - 1);
      indicator_group_NA_remove = indicator_group.rows(find_finite(indicator_group.col(0)));
      group0 = find(sum(indicator_group_NA_remove,0) == 1);
      
      indicator_group = as<mat>(transformation_indicator[ell]).rows(as<uvec>(group[1]) - 1);
      indicator_group_NA_remove = indicator_group.rows(find_finite(indicator_group.col(0)));
      group1 = find(sum(indicator_group_NA_remove,0) == 1);
      
      double ta_gamma;
      double tb_gamma;
      int K1 = as<uvec>(group[0]).n_elem;
      ta_gamma = a_gamma[ell] + group0.size() * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(group0)%as<mat>(mu[ell]).cols(group0))/2;
      // cout << "ta_gamma -- " << ta_gamma << endl;
      // cout << "tb_gamma -- " << tb_gamma << endl;
      double sgga2_grp1 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      int K2 = as<uvec>(group[1]).n_elem;
      ta_gamma = a_gamma[ell] + group1.size() * p[ell] / 2;
      tb_gamma = b_gamma[ell] + accu(as<mat>(mu[ell]).cols(group1)%as<mat>(mu[ell]).cols(group1))/2;
      // cout << "ta_gamma -- " << ta_gamma << endl;
      // cout << "tb_gamma -- " << tb_gamma << endl;
      double sgga2_grp2 = 1 / randg<double>(distr_param(ta_gamma, 1/tb_gamma));
      vec sgga2_ell = as<vec>(sgga2[ell]);
      for (int k = 0; k < K1; k++) {
        sgga2_ell[k] = sgga2_grp1;
      }
      for (int k = 0; k < K2; k++) {
        sgga2_ell[k+K1] = sgga2_grp2;
      }
      sgga2[ell] = sgga2_ell;

      // ----------------------- //
      // Update sgbeta2
      // ----------------------- //
      double ta_beta = a_beta[ell] + K[ell] / 2;
      double tb_beta = b_beta[ell] + accu(as<mat>(beta0[ell])%as<mat>(beta0[ell]))/2;
      // sgbeta2[ell] = tb_beta/ta_beta;
      
      // cout << "ta_beta -- " << ta_beta << endl;
      // cout << "tb_beta -- " << tb_beta << endl;
      sgbeta2[ell] = (1 / randg<double>(distr_param(ta_beta, 1/tb_beta)));
      if(sgbeta2[ell] > 1e2) {
        sgbeta2[ell] = 1e2;
      }
      // ----------------------- //
      // Update omega
      // ----------------------- //
      for (int k = 0; k < K[ell]; k++) {
        double at = as<mat>(a[ell])[k] + as<mat>(Delta[ell])[k];
        double bt = as<mat>(b[ell])[k] + (1 - as<mat>(Delta[ell])[k]);
        mat omega_ell = as<mat>(omega[ell]);
        
        omega_ell[k] = R::rbeta(at,bt);
        omega[ell] = omega_ell;
      }
      
      if(iter >= (int)burnin){
        if((iter - burnin) % thin == 0){
          as<List>(beta0res[ell])[l] = as<mat>(transformation_indicator[ell]) * as<mat>(beta0[ell]);
          as<List>(Sgga2Res[ell])[l] = sgga2[ell];
          as<List>(Sgal2Res[ell])[l] = sgal2xi2[ell];
          as<List>(Sgbeta2Res[ell])[l] = sgbeta2[ell];
          as<List>(omegaRes[ell])[l] = as<mat>(transformation_indicator[ell]) * as<mat>(omega[ell]);
          as<List>(DeltaRes[ell])[l] = as<mat>(transformation_indicator[ell]) * as<mat>(Delta[ell]);
          as<List>(mutRes[ell])[l] = mut[ell];
          as<List>(muRes[ell])[l] = mu[ell];
          as<List>(QRes[ell])[l] = VarCompTest_out["Q"];
          as<List>(WRes[ell])[l] = VarCompTest_out["W"];
        }
      }
    }
    
    List omega_full_dim(L);
    for (ell = 0; ell < L; ell++) {
      omega_full_dim[ell] = as<mat>(transformation_indicator[ell]) * as<mat>(omega[ell]);
    }
    mat alpha_all = do_call_rbind_vecs(omega_full_dim);
    mat U = log(alpha_all / (1 - alpha_all)) - u0;
    U = up_truncate_matrix(U);
    U = low_truncate_matrix(U);

    // impute
    mat missing_status = U;
    missing_status = missing_status * 0 + 1;
    
    // cout << iter << endl;
    // cout << "U -- " << U << endl;
    // cout << "Start imputation ..." << endl;
    if(missing_method == "MIDAS") {
      if(fast_impute) {
        if(iter <= 1) {
          U = cppMIDAS(U);
          U_complete = U;
        } else {
          U = as<mat>(fillNA(U,U_complete));
        }
      } else {
        U = cppMIDAS(U);
      }
    } else if (missing_method == "missForest") {
      if(fast_impute) {
        if(iter <= 1) {
          U = cppmissForest(U);
          U_complete = U;
        } else {
          U = as<mat>(fillNA(U,U_complete));
        }
      } else {
        U = cppmissForest(U);
      }
    }
    // cout << "End imputation ..." << endl;
    // cout << "U -- " << U << endl;

    // MVL
    // split for MVL
    mat U1 = U.cols(as<uvec>(group[0]) - 1);
    mat U2 = U.cols(as<uvec>(group[1]) - 1);
    if (mvl_method == "CCAPCA"){
      List ccas = Rcpp::as<List>(cc(U1, U2));
      mat Ahat = ccas["xcoef"];
      mat Bhat = ccas["ycoef"];
      mat XX = U1 * Ahat;
      mat YY = U2 * Bhat;
      mat X_est1 = XX.cols(0, CC - 1) * pinv(Ahat.cols(0, CC - 1));
      mat X_est2 = YY.cols(0, CC - 1) * pinv(Bhat.cols(0, CC - 1));
      
      mat X_res1 = U1 - X_est1;
      mat X_res2 = U2 - X_est2;
      mat U;
      vec s;
      mat V;
      mat norm_U1 = normalize_mat(X_res1);
      mat norm_U2 = normalize_mat(X_res2);
      
      mat colmean1 = mean(X_res1, 0);
      mat colmean2 = mean(X_res2, 0);
      mat sd1 = stddev(X_res1, 0, 0);
      mat sd2 = stddev(X_res2, 0, 0);
      
      svd(U, s, V, norm_U1);
      mat X_red1 = U.cols(0, PC1 - 1) * diagmat(s.subvec(0, PC1 - 1)) * trans(V.cols(0, PC1 - 1));
      
      svd(U, s, V, normalize_mat(X_res2));
      mat X_red2 = U.cols(0, PC2 - 1) * diagmat(s.subvec(0, PC2 - 1)) * trans(V.cols(0, PC2 - 1));
      
      X_red1 = X_red1 * diagmat(sd1);
      for (int j = 0; j < X_red1.n_cols; j++) {
        X_red1.col(j) += colmean1[j];
      }
      X_red2 = X_red2 * diagmat(sd2);
      for (int j = 0; j < X_red2.n_cols; j++) {
        X_red2.col(j) += colmean2[j];
      }
      
      U1 = X_est1 + X_red1;
      U2 = X_est2 + X_red2;
    } else {
      List res = DeepCCA(U1, U2, Named("method") = mvl_method, Named("LATENT_DIMS") = latent_dim, Named("EPOCHS") = epochs, Named("nw") = 0);
      U1 = as<mat>(res[0]);
      U2 = as<mat>(res[1]);
    }

    mat U_est = join_rows(U1, U2);
    U_est = U_est % missing_status;
    for (int ell = 0; ell < L; ell++) {
      mat current_omega;
      mat U_ell = U_est.row(ell);
      U_ell = U_ell.elem(find_finite(U_ell));
      current_omega = 1 / (1 + exp(- U_ell - u0));
      // cout << "current_omega" << current_omega << endl;
      omega[ell] = (current_omega);
    }
    if(iter >= (int)burnin){
      if((iter - burnin) % thin == 0){
        l += 1;
      }
    }
  }
  
  List res = List::create(
    _["beta0res"] = beta0res,
    _["Sgga2Res"] = Sgga2Res,
    _["Sgal2Res"] = Sgal2Res,
    _["Sgbeta2Res"] = Sgbeta2Res,
    _["omegaRes"] = omegaRes,
    _["DeltaRes"] = DeltaRes,
    _["mutRes"] = mutRes,
    _["muRes"] = muRes,
    _["QRes"] = QRes,
    _["WRes"] = WRes,
    _["PRes"] = PRes
  );
  return res;
}


// [[Rcpp::export]]
List mintMR(List gammah, const List Gammah,
            List se1, const List se2,
            Nullable<List> group = R_NilValue,
            Nullable<List> opts = R_NilValue,
            Nullable<List> corr_mat = R_NilValue,
            Nullable<arma::mat> reference = R_NilValue,
            Nullable<arma::mat> Lambda = R_NilValue,
            int CC = 2, int PC1 = 1, int PC2 = 1,
            int latent_dim = 2, 
            bool display_progress = true,
            String missing_method = "missForest",
            String mvl_method = "DVCCA",
            int epochs = 5) {
  int L = gammah.size();
  List corr_mat_list(L), opts_list(L);
  bool overlapped = Lambda.isNotNull();
  bool missingness;
  
  if (accu(check_missing(gammah)) > 0){
    missingness = true;
  }
  
  if (opts.isNull()) {
    opts_list = get_opts(L);
  } else {
    opts_list = as<List>(opts.get());
  }
  
  if (corr_mat.isNull()) {
    for (int i = 0; i < L; ++i) {
      int n = as<mat>(gammah[i]).n_rows;
      corr_mat_list[i] = eye<mat>(n, n);
    }
  } else {
    corr_mat_list = as<List>(corr_mat.get());
  }
  
  List res, group_list;
  
  arma::mat lambda_mat;
  if(overlapped){
    lambda_mat = as<arma::mat>(Lambda.get());
  } else {
    int n = as<mat>(gammah[0]).n_cols;
    lambda_mat = eye<mat>(n+1, n+1);
  }
  if(!missingness){
    if(reference.isNull()){
      if(group.isNull()){
        res= mintMR_single_omics(gammah, Gammah, se1, se2, corr_mat_list, opts_list, lambda_mat, display_progress, PC1);
      } else {
        group_list = as<List>(group.get());
        res = mintMR_multi_omics(gammah, Gammah, se1, se2, corr_mat_list, group_list, opts_list, lambda_mat, display_progress, CC, PC1, PC2);
      }
    } else {
      mat reference_m = as<mat>(reference.get());
      res = mintMR_single_omics_supervised(gammah, Gammah, se1, se2, reference_m, corr_mat_list, opts_list, lambda_mat, display_progress, PC1);
    }
  } else {
    if(missingness){
      if(!group.isNull()){
        group_list = as<List>(group.get());
        res = mintMR_Impute_MVL(gammah, Gammah, se1, se2, corr_mat_list, group_list, 
                                opts_list, lambda_mat, display_progress, 
                                latent_dim, CC, PC1, PC2, missing_method, mvl_method, epochs);
      }
    }
  }


  List VCP = vc_test(res);
  List summary = summarize_result(res);
  // return summary;
  return List::create(Named("Pvalue") = summary["Pvalue"],
                      Named("Estimate") = summary["Estimate"],
                                                 Named("VCP") = VCP,
                                                 Named("res") = res);
}
