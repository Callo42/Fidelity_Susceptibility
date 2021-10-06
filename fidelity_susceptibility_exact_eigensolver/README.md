# 使用Exact Eigensolver求解

- 此目录下使用Lanczos辅助求解对角化问题，其中对矩阵进行了显式构造
- 优化使用了lax.stop_gradient，可以避免在求解过程中两次对角化
- 尝试了对于矩阵-向量相乘方法进行实现，但未完成，转移到```chiF_Hv_product```中进行

