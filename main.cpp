#include <iostream>
#include <vector>

#define ARMA_ALLOW_FAKE_GCC before
#include <armadillo>

using namespace arma;

/**
 * Convert an std::vector to Armadillo vector.
 * It assumes v_arma is preallocated.
 * 
 * @param v the std::vector
 * @param[out] v_arma the Armadillo vector
 */
void DoubleVectorToArmaVec(std::vector<double>& v, vec& v_arma)
{
  for (size_t i = 0; i < v.size(); i++)
  {
    v_arma(i) = v[i];
  }
}

/**
 * Convert an std::vector of std::vector(s) to Armadillo matrix.
 * It assumes A_arma is preallocated.
 * 
 * @param A the std::vector of std::vector(s)
 * @param[out] A_arma the Armadillo matrix
 */
void DoubleVector2DToArmaMat(std::vector<std::vector<double>>& A, mat& A_arma)
{
  for (size_t i = 0; i < A.size(); i++)
  {
    for (size_t j = 0; j < A[i].size(); j++)
    {
      A_arma(i, j) = A[i][j];
    }
  }
}

/**
 * Convert an Armadillo vector to std::vector.
 * @param v_arma the Armadillo vector
 * @param[out] v the std::vector
 */
void ArmaVecToDoubleVector(vec& v_arma, std::vector<double>& v)
{
  // if v preallocated
  if (v_arma.n_rows == v.size())
  {
    for (size_t i = 0; i < v_arma.n_rows; i++)
    {
      v[i] = v_arma(i);
    }
  }
  else
  {
    for (size_t i = 0; i < v_arma.n_rows; i++)
    {
      v.push_back(v_arma(i));
    }
  }
}

int main(int argc, char** argv)
{
  // build coefficient matrix
  std::vector<std::vector<double>> A;
  std::vector<double> row1{1.0, 2.0, 3.0};
  std::vector<double> row2{4.0, 5.0, 6.0};
  std::vector<double> row3{7.0, 8.0, 9.0};
  A.push_back(row1);
  A.push_back(row2);
  A.push_back(row3);

  // print coefficient matrix to standard output
  std::cout << "Initial coefficient matrix\n";
  std::cout << "Rows: " << A.size() << "\n";
  std::cout << "Columns: " << A[0].size() << "\n";
  for (size_t i = 0; i < A.size(); i++)
  {
    for (size_t j = 0; j < A[i].size(); j++)
    {
      std::cout << A[i][j] << " ";
    }
    std::cout << "\n";
  }

  // build right-hand side vector
  std::vector<double> b{10.0, 11.0, 12.0};

  // print right-hand side vector to standard output
  std::cout << "Initial right-hand side vector\n";
  std::cout << "Rows: " << b.size() << "\n";
  for (size_t i = 0; i < b.size(); i++)
  {
    std::cout << b[i] << "\n";
  }

  // convert to Armadillo data structures
  mat A_arma(A.size(), A[0].size(), fill::zeros);
  DoubleVector2DToArmaMat(A, A_arma);
  vec b_arma(b.size(), fill::zeros);
  DoubleVectorToArmaVec(b, b_arma);

  std::cout << "--------------------------------------------------\n";

  // print Armadillo coefficient matrix to standard output
  std::cout << "Armadillo coefficient matrix:\n";
  std::cout << "Rows: " << A_arma.n_rows << "\n";
  std::cout << "Columns: " << A_arma.n_cols << "\n";
  for (size_t i = 0; i < A_arma.n_rows; i++)
  {
    for (size_t j = 0; j < A_arma.n_cols; j++)
    {
      std::cout << A_arma(i, j) << " ";
    }
    std::cout << "\n";
  }

  // print Armadillo right-hand side vector to standard output
  std::cout << "Armadillo right-hand side vector:\n";
  std::cout << "Rows: " << b_arma.n_rows << "\n";
  for (size_t i = 0; i < b_arma.n_rows; i++)
  {
    std::cout << b_arma(i) << "\n";
  }

  // solve dense least squares
  vec x_arma = solve(A_arma, b_arma, solve_opts::force_approx);

  // print Armadillo solution vector to standard output
  std::cout << "Armadillo solution vector:\n";
  std::cout << "Rows: " << x_arma.n_rows << "\n";
  for (size_t j = 0; j < x_arma.n_rows; j++)
  {
    std::cout << x_arma(j) << "\n";
  }

  std::cout << "--------------------------------------------------\n";

  // convert Armadillo solution vector to std::vector
  std::vector<double> x;
  ArmaVecToDoubleVector(x_arma, x);

  // print solution vector to standard output
  std::cout << "Solution vector\n";
  std::cout << "Columns: " << x.size() << "\n";
  for (size_t i = 0; i < x.size(); i++)
  {
    std::cout << x[i] << "\n";
  }

  return 0;
}