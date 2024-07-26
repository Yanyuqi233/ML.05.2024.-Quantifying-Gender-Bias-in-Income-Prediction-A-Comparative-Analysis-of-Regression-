# (Machine Leaning 05/2024) Quantifying Gender Bias in Income Prediction: A Comparative Analysis of Regression Models

## Goal
This study focuses on quantifying **the impact of gender on income prediction by comparing the predictive quality of popular regression models when gender data is or isn’t available**. 
Using a dataset from the CPS(Current Population Survey), and employing regression analysis to predict income levels.

The research contributes to the ongoing dialogue on gender equity and underscores the importance of fair and unbiased predictive modeling practices in tackling gender-based wage inequalities.

## Methods
### Regression Models (Analysis of data from 2023-2013)
<div align="center">
  <img src="./Images/Model Morphisms.png" alt=" Regression Model Morphisms" width="50%" height="auto">
</div>

### Clustering (Analysis of data from 2023)
The report utilizes the **HDBSCAN clustering algorithm**, which is a density-based clustering method particularly suited for high-dimensional and large datasets.

## Result
### Model Performance
The **Lasso, Ridge, and simple neural network architectures exhibited enhanced performance metrics** when gender was included in the feature set. Across different evaluation criteria, such as mean squared error and mean absolute error, these models consistently demonstrated lower error rates, indicative of their heightened accuracy in income prediction tasks.

### Inference Performance
The report used regression models to observe that predicted income levels consistently favored one gender across all cases, indicating a systemic bias. This bias likely stems from the gender-related feature weights in the models and the regression models' structure. The findings provide clear evidence of persistent disparities in income predictions based on gender, advancing the understanding of **quantifiable gender bias**.

### Clustering
In 2023, the position of first-line supervisors of retail sales workers has a higher proportion of females. In Driver/sales workers and truck drivers, males have a higher proportion. Females make up a higher proportion of registered nurses and cashiers.

In 2023, in the fields of First-Line supervisors of retail sales workers, Driver/sales workers and truck drivers, Registered nurses, and Cashiers, there is little difference in income between men and women.

## Insight and Conclusion
* Gender is a crucial input feature.
* There is an inherent bias in the model predictions.
* This study advances gender bias quantification.
* PromisingProspectsforGenderEquality.

The research provides a comprehensive analysis of gender's impact on income prediction and highlights its importance for reducing workplace disparities by demonstrating how gender significantly influences model accuracy when included as a feature. Our causality investigations during inference reveal inherent biases in models, leading to systematic differences in income predictions based on gender, underscoring the need for robust mitigation strategies to ensure fairness. Despite ongoing challenges, the presence of smaller gender wage disparities in specific sectors in 2023 offers hope for achievable gender wage equality through sustained research and targeted policies. Our study contributes to discussions on gender equity by quantifying bias and its effects on income prediction, advocating for inclusive and equitable modeling practices, and emphasizing the necessity of ongoing efforts to address disparities and advance gender equality in the workplace.
