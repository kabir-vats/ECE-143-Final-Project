# ECE-143-Final-Project

## Classification with basic machine learning approaches

While deep learning models are widely used in text classification tasks and have demonstrated strong performance, traditional machine learning methods offer distinct advantages, particularly in resource-constrained environments.

In this section, we implemented and evaluated the following traditional machine learning models for fake news classification:  

- **Naïve Bayes (MultinomialNB)**  
- **Logistic Regression**  
- **Support Vector Machine (SVM) with Linear Kernel**  
- **Random Forest Classifier**  
- **K-Nearest Neighbors (KNN)**  

For text representation, the following tokenization methods were applied:

- **TF-IDF vectorization**
- **Word Count vectorization**

To ensure modularity and ease of experimentation, both the model and tokenizer were integrated into a unified pipeline, enabling seamless training and evaluation across different configurations.