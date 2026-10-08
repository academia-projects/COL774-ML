# Oblique Decision Tree

In the oblique decision tree that we have made, we applied logistic regression with L2 regularization to find the node split. The reason for using logistic regression with L2 regularization is to prevent overfitting and to ensure that the model generalizes well to unseen data.

We also found the optimal hyperparameter C, which came out to be 1.3. This value of C balances the trade-off between achieving a low training error and maintaining a low regularization strength.

Additionally, we tried different threshold calculating objective functions such as cross-entropy, Gini index, and misclassification rate to evaluate the performance of the splits.

# Summary of Methods and Findings

- **Logistic Regression with L2 Regularization**: Used to find node splits to prevent overfitting.
- **Optimal Hyperparameter C**: 1.3
- **Max Depth**: 10
- **Objective Functions Tested**:
    - Cross-Entropy
    - Gini Index
    - Misclassification Rate
    Finally used Gini Index, as it gave the best results.

These methods and findings helped in constructing a robust oblique decision tree model.

# Additional Insights

In our experiments, we also included bias terms for the decision boundary at every node. However, this did not lead to better results. One possible reason for this could be that while the bias term might help in finding the best boundary for a single node, it does not necessarily contribute to the overall performance of the model. For a more effective model, it is important to have a series of good boundaries across multiple nodes rather than the best boundary at just one node.

# Conclusion

The methods and findings discussed above were instrumental in constructing a robust oblique decision tree model. Despite the inclusion of bias terms not improving the results, the use of logistic regression with L2 regularization and the evaluation of different objective functions provided a strong foundation for the model.
