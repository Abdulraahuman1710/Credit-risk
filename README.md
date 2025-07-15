Credit Risk Classification Model
Project Overview
This project aims to develop a Credit Risk Classification Model to predict the likelihood of credit default. The model uses a dataset of financial features and outcomes, providing insights into creditworthiness and aiding decision-making processes.

Key Highlights
Model Used: Logistic Regression, Class Imbalance Handling: SMOTE Tomek, Hyperparameter Tuning: Optuna

Performance Metrics
1. Decile Analysis
Top Deciles: Decile 9: Event rate of 72.00%, non-event rate of 28.00%. The model is highly confident in predicting events. Decile 8: Event rate of 12.72%, cumulative event rate reaching 98.6%. Middle Deciles: Deciles 7 and 6 show a significant drop in event rates. Lower Deciles: Deciles 5 to 0 have 0% events, indicating all predictions are non-events.

2. KS Statistic
Highest KS Value: 85.98% at Decile 8, showing the model performs best at distinguishing events and non-events in this range. A KS value in the top 3 deciles and above 40% indicates a good predictive model.

3. AUC & Gini Coefficient
AUC: 0.98 - Excellent at distinguishing events from non-events. Gini Coefficient: 0.96 - Indicates almost perfect rank-ordering capability.

Tools & Techniques Used
Model Training: Logistic Regression Imbalance Handling: SMOTE Tomek Hyperparameter Optimization: Optuna Evaluation Metrics: KS Statistic, AUC, Gini Coefficient, Decile Analysis

Conclusion
The model is highly effective in predicting credit risk, demonstrated by:

High AUC (0.98) and Gini (0.96) scores. Robust performance in top deciles with significant event rates. Strong KS statistic in top deciles, meeting industry benchmarks for a good predictive model.

Future Work
Extend model to include additional features for enhanced predictions. Deploy the model for real-time credit risk assessments. Explore ensemble methods to further improve classification accuracy.
