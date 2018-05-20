#ifndef TypeDefs_H
#define TypeDefs_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using Eigen::SparseVector;

enum EnergyMethod {FULL, PCR, AN08, PRED_WEIGHTS_L1, PRED_DIRECT};

#endif