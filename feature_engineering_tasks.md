# Feature Engineering Task List

1. **Generate new features** Create new variables by combining or transforming existing ones.

2. **Discretization (Binning)** Convert continuous variables into categorical bins if needed.

3. **Encoding (Ordinal, One-hot, etc.)** Transform categorical variables into numerical formats.

4. **Transformation (Logarithmic, Exponential, Square Root, Box–Cox, Yeo–Johnson, etc.)** Apply mathematical transformations to normalize or rescale distributions.

5. **Scaling (Normalization, Standardization, Min–Max, etc.)** Scale numerical variables to a consistent range.

6. **Feature Selection**

   - **Variance Threshold:** Remove features with low variance.
   - **Correlation:** Eliminate highly correlated (redundant) features.
   - **Chi-squared, ANOVA:** Select features relevant to the target using statistical tests.

7. **Dimensionality Reduction**
   - **Principal Component Analysis (PCA):** Reduce feature space while retaining variance.
   - **Factor Analysis (FA):** Identify underlying latent variables if required.
