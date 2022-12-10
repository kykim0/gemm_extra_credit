task void gemm_ispc_task(uniform int idxX,
                         uniform int idxY,
                         uniform int lengthX,
                         uniform int lengthY,
                         uniform int m,
                         uniform int n,
                         uniform int k,
                         uniform double A[],
                         uniform double B[],
                         uniform double C[],
                         uniform double alpha,
                         uniform double beta) {
  uniform int startX = idxX * lengthX;
  uniform int startY = idxY * lengthY;
  foreach (i = startX ... startX + lengthX,
           j = startY ... startY + lengthY) {
    double inner_prod = 0.0;
    for (uniform int kk = 0; kk < k; ++kk) {
      inner_prod += A[i*k+kk] * B[kk*n+j];
    }
    C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
  }
}

export void gemm_ispc(uniform int m,
                      uniform int n,
                      uniform int k,
                      uniform double A[],
                      uniform double B[],
                      uniform double C[],
                      uniform double alpha,
                      uniform double beta) {
	// YOUR IMPLEMENTATION HERE
  uniform int numPartsX = 8;
  uniform int numPartsY = 16;
  uniform int lengthX = m / numPartsX;
  uniform int lengthY = n / numPartsY;
  for (uniform int idx = 0; idx < numPartsX * numPartsY; ++idx) {
    uniform int idxX = (idx / numPartsY);
    uniform int idxY = (idx % numPartsY);
    launch gemm_ispc_task(idxX, idxY, lengthX, lengthY,
                          m, n, k, A, B, C, alpha, beta);
  }
}
