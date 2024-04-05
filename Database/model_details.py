# Dictionary containing model information
models_info = {

    'Decision Tree': {
        'Description': ["\n","Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation."
        ],
        'Type': ['Supervised Learning'
        ],
        'SubType': ['Classification'

        ],
        'Usage Scenarios': ["\n",
            "- Classification tasks with non-linear decision boundaries.",
            "- Regression tasks with non-linear relationships between features and target variable."
        ],
        'Advantages': ["\n",
            "- Simple and easy to interpret.",
            "- Handles both numerical and categorical data.",
            "- Robust to outliers and irrelevant features."
        ],
        'Limitations': ["\n",
            "- Prone to overfitting, especially with deep trees.",
            "- Sensitive to small variations in the training data.",
            "- Not suitable for capturing complex interactions between features."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/Decision%20Tree/DecisionTree.ipynb'
        ],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/Decision%20Tree/DecisionTree.html"
        ],
        'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
        ],
        'Implementation Code': (
            "from sklearn.tree import DecisionTreeClassifier\n"
            "\n"
            "# Instantiate Decision Tree classifier\n"
            "model = DecisionTreeClassifier()\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }
}


models_info['Linear Regression'] = {
    'Description': ["\n",
        "Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the input features and the target variable."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Regression'
    ],
    'Usage Scenarios': ["\n",
        "- Predicting continuous target variables.",
        "- Understanding the relationship between independent and dependent variables."
    ],
    'Advantages': ["\n",
        "- Simple and easy to interpret.",
        "- Can handle large datasets efficiently.",
        "- Provides insights into the relationship between variables."
    ],
    'Limitations': ["\n",
        "- Assumes a linear relationship between variables, which may not always hold true.",
        "- Sensitive to outliers and multicollinearity.",
        "- May not capture complex non-linear relationships."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be numeric.",
        "- X and y should have consistent lengths."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Regression/LinearRegression/LinearRegression.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Regression/LinearRegression/LinearRegression.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
    ],
    'Implementation Code': (
        "from sklearn.linear_model import LinearRegression\n"
        "\n"
        "# Instantiate Linear Regression model\n"
        "model = LinearRegression()\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict target values for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "score = model.score(X_test, y_test)\n"
        "print('R^2 Score:', score)"
    )
}


models_info['Ridge Regression'] = {
    'Description': ["\n",
        "Ridge Regression is a linear regression technique that adds a penalty term to the ordinary least squares (OLS) method. This penalty term helps to regularize the coefficients, preventing overfitting by shrinking them towards zero."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Regression'
    ],
    'Usage Scenarios': ["\n",
        "- Dealing with multicollinearity in the dataset.",
        "- Preventing overfitting in linear regression models.",
        "- Handling situations where the number of features exceeds the number of samples."
    ],
    'Advantages': ["\n",
        "- Helps in reducing overfitting by adding a penalty term.",
        "- Works well in situations with multicollinearity.",
        "- Can handle high-dimensional datasets efficiently."
    ],
    'Limitations': ["\n",
        "- Requires tuning of the regularization parameter (alpha).",
        "- May not perform well if the number of features is much greater than the number of samples.",
        "- Assumes a linear relationship between variables."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be numeric.",
        "- X and y should have consistent lengths."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Regression/RidgeRegression/RidgeRegression.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Regression/RidgeRegression/RidgeRegression.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"
    ],
    'Implementation Code': (
        "from sklearn.linear_model import Ridge\n"
        "\n"
        "# Instantiate Ridge Regression model\n"
        "model = Ridge(alpha=1.0)  # alpha is the regularization strength\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict target values for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "score = model.score(X_test, y_test)\n"
        "print('R^2 Score:', score)"
    )
}

models_info['Lasso Regression'] = {
    'Description': ["\n",
        "Lasso Regression is a linear regression technique that adds a penalty term to the ordinary least squares (OLS) method, similar to Ridge Regression. However, Lasso Regression uses L1 regularization, which tends to shrink some coefficients to exactly zero. This makes Lasso Regression useful for feature selection, as it can automatically eliminate irrelevant features."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Regression'
    ],
    'Usage Scenarios': ["\n",
        "- Feature selection: Lasso Regression can be used when there's a need to identify the most important features in a dataset.",
        "- Dealing with high-dimensional datasets: It performs well when the number of features is much greater than the number of samples.",
        "- Situations where interpretability is important: Lasso can help by producing sparse models with fewer non-zero coefficients."
    ],
    'Advantages': ["\n",
        "- Automatic feature selection: Lasso can select the most relevant features by setting some coefficients to zero.",
        "- Handles multicollinearity: Lasso can handle situations where features are highly correlated.",
        "- Produces interpretable models: Sparse models with fewer non-zero coefficients are easier to interpret."
    ],
    'Limitations': ["\n",
        "- Sensitive to the scaling of features: Features should be scaled before applying Lasso Regression.",
        "- Requires tuning of the regularization parameter (alpha).",
        "- Assumes a linear relationship between variables."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be numeric.",
        "- X and y should have consistent lengths."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Regression/LassoRegression/LassoRegression.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Regression/LassoRegression/LassoRegression.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"
    ],
    'Implementation Code': (
        "from sklearn.linear_model import Lasso\n"
        "\n"
        "# Instantiate Lasso Regression model\n"
        "model = Lasso(alpha=1.0)  # alpha is the regularization strength\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict target values for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "score = model.score(X_test, y_test)\n"
        "print('R^2 Score:', score)"
    )
}

models_info['MLPRegressor'] = {
    'Description': ["\n",
        "MLPRegressor, short for Multi-layer Perceptron Regressor, is a type of artificial neural network model used for regression tasks. It consists of multiple layers of nodes, including an input layer, one or more hidden layers, and an output layer. Each node applies a non-linear activation function to the weighted sum of its inputs. MLPRegressor can learn complex relationships between inputs and outputs and is capable of approximating any continuous function given enough nodes and layers."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Regression | Neural Networks'
    ],
    'Usage Scenarios': ["\n",
        "- Predicting continuous target variables: MLPRegressor is suitable for regression tasks where the target variable is continuous.",
        "- Non-linear relationships: It can capture non-linear relationships between features and the target variable.",
        "- Handling complex data: MLPRegressor can handle high-dimensional data and large datasets effectively."
    ],
    'Advantages': ["\n",
        "- Capability to model complex relationships: MLPRegressor can approximate complex functions due to its non-linear activation functions and multiple layers.",
        "- Flexibility: It can handle various types of data and problem domains.",
        "- Robustness: MLPRegressor can handle noisy data and adapt to different distributions."
    ],
    'Limitations': ["\n",
        "- Requires careful tuning of hyperparameters: Proper tuning of parameters such as the number of layers, number of nodes per layer, and learning rate is essential for optimal performance.",
        "- Susceptible to overfitting: MLPRegressor can overfit to the training data, especially with large networks and insufficient regularization.",
        "- Computationally expensive: Training MLPRegressor models can be computationally intensive, particularly for large datasets or complex architectures."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be numeric.",
        "- X and y should have consistent lengths."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Regression/NeuralNetworksRegression/NeuralNetworkRegressor.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Regression/NeuralNetworksRegression/NeuralNetworkRegressor.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"
    ],
    'Implementation Code': (
        "from sklearn.neural_network import MLPRegressor\n"
        "\n"
        "# Instantiate MLPRegressor model\n"
        "model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,\n "
        "                      batch_size='auto', learning_rate='constant', learning_rate_init=0.001,\n "
        "                      max_iter=200, shuffle=True, random_state=None, tol=0.0001,\n "
        "                      verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,\n "
        "                      early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,\n "
        "                      epsilon=1e-08, n_iter_no_change=10, max_fun=15000)\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict target values for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "score = model.score(X_test, y_test)\n"
        "print('R^2 Score:', score)"
    )
}

models_info['Random Forest'] = {
    'Description': ["\n",
        "Random Forest Classification is an ensemble learning technique used for classification tasks. It builds multiple decision trees during training and outputs the mode (classification) of the individual trees as the final prediction. Random forests introduce randomness during the training process, both in the selection of data points used to build each tree and the features considered for splitting at each node. This randomness helps to reduce overfitting and improve generalization."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Classification | Ensemble Learning'
    ],
    'Usage Scenarios': ["\n",
        "- Predicting categorical target variables: Random Forest Classification is suitable for classification tasks where the target variable is categorical.",
        "- Handling imbalanced datasets: It can handle imbalanced datasets well by using the class weights parameter.",
        "- Dealing with high-dimensional data: Random Forest Classification can handle high-dimensional data and large datasets effectively."
    ],
    'Advantages': ["\n",
        "- Robustness to overfitting: Random forests are less prone to overfitting compared to individual decision trees, thanks to ensemble averaging and feature randomness.",
        "- Handles missing values and outliers: Random forests can handle missing values and outliers in the data without requiring preprocessing.",
        "- Provides feature importances: Random forests can indicate the relative importance of features in predicting the target variable."
    ],
    'Limitations': ["\n",
        "- Lack of interpretability: While random forests provide accurate predictions, the individual trees' decision-making process can be hard to interpret.",
        "- Computational complexity: Training and prediction with random forests can be computationally expensive, especially for large datasets and many trees.",
        "- Less effective on noisy data: Random forests may not perform well on noisy data or data with irrelevant features."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be categorical (class labels).",
        "- X and y should have consistent lengths."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Classification/RandomForest/RandomForest.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/RandomForest/RandomForest.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
    ],
    'Implementation Code': (
        "from sklearn.ensemble import RandomForestClassifier\n"
        "\n"
        "# Instantiate Random Forest Classification model\n"
        "model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,\n "
        "                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',\n "
        "                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,\n "
        "                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None,\n "
        "                                verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,\n "
        "                                max_samples=None)\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict class labels for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "accuracy = model.score(X_test, y_test)\n"
        "print('Accuracy:', accuracy)"
    )
}


# Add entry for Logistic Regression
models_info['Logistic Regression'] = {
    'Description': ["\n",
        "Logistic Regression is a linear model for binary classification that predicts the probability of occurrence of an event by fitting data to a logistic curve. Despite its name, it's used for classification rather than regression. It works by modeling the probability that each input belongs to a particular category."
    ],
    'Type': ['Supervised Learning'
    ],
    'SubType': ['Classification'
    ],
    'Usage Scenarios': ["\n",
        "- Binary classification tasks.",
        "- Probability estimation tasks."
    ],
    'Advantages': ["\n",
        "- Simple and efficient for binary classification.",
        "- Outputs well-calibrated predicted probabilities.",
        "- Interpretable coefficients."
    ],
    'Limitations': ["\n",
        "- Assumes a linear relationship between features and log-odds of the outcome.",
        "- May not perform well with high-dimensional features or correlated features."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y should be binary (0 or 1) for binary classification."
    ],
    'Example': ['https://github.com/BAProjects/ML_Library/blob/main/Classification/LogisticRegression/LogisticRegression.ipynb'
    ],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/LogisticRegression/LogisticRegression.html"
    ],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    ],
    'Implementation Code': (
        "from sklearn.linear_model import LogisticRegression\n"
        "\n"
        "# Instantiate Logistic Regression classifier\n"
        "model = LogisticRegression()\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict labels for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "accuracy = model.score(X_test, y_test)\n"
        "print('Accuracy:', accuracy)"
    )
}

# Dictionary containing model information


models_info['Naive Bayes'] = {
    'Description': ["\n",
        '''Naive Bayes is a family of probabilistic algorithms based on Bayes' Theorem,
          which assumes independence between features. Despite its simplicity, Naive Bayes is known to perform well in many real-world situations, 
          particularly in text classification and spam filtering. Naive Bayes calculates the probability of whether a data point belongs within a certain category or does not. '''
    ],
    'Type': ['Supervised Learning'],
    'SubType': ['Classification'],
    'Usage Scenarios': ["\n",
        "- Text classification tasks such as spam detection, sentiment analysis, and document categorization.",
        "- Real-time prediction tasks due to its simplicity and speed."
    ],
    'Advantages': ["\n",
        "- Simple and easy to implement.",
        "- Requires a small amount of training data to estimate the parameters.",
        "- Highly scalable and suitable for large datasets.",
        "- Performs well in the presence of irrelevant features."
    ],
    'Limitations': ["\n",
        "- Assumes independence between features, which might not hold true in real-world datasets.",
        "- Can be outperformed by more complex models when the independence assumption is violated.",
        "- Sensitivity to the scale of numerical features."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X should be numeric or preprocessed to numeric form.",
        "- y can be numeric or string.",
        "- X and y should have consistent lengths."
    ],
    'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/NaiveBayes/NaiveBayes.ipynb'],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/NaiveBayes/NaiveBayes.html"],
    'Documentation': ["https://scikit-learn.org/stable/modules/naive_bayes.html"],
    'Implementation Code': (
        "from sklearn.naive_bayes import GaussianNB\n"
        "\n"
        "# Instantiate Gaussian Naive Bayes classifier\n"
        "model = GaussianNB()\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict labels for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "accuracy = model.score(X_test, y_test)\n"
        "print('Accuracy:', accuracy)"
    )
}


# Dictionary containing model information


models_info['Support Vector Machine'] = {
        'Description': ["\n",
            "Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space. SVM can handle both linear and non-linear data by using appropriate kernel functions."
        ],
        'Type': ['Supervised Learning'],
        'SubType': ['Classification'],
        'Usage Scenarios': ["\n",
            "- Classification tasks with complex decision boundaries and high-dimensional feature spaces.",
            "- Regression tasks with non-linear relationships between features and target variable.",
            "- Outlier detection."
        ],
        'Advantages': ["\n",
            "- Effective in high-dimensional spaces.",
            "- Versatile: can handle linear and non-linear data using different kernel functions.",
            "- Robust against overfitting, especially in high-dimensional space.",
            "- Memory efficient as it uses a subset of training points (support vectors) in the decision function."
        ],
        'Limitations': ["\n",
            "- SVMs are memory intensive and may be slow to train on large datasets.",
            "- Choosing the appropriate kernel function and tuning hyperparameters can be challenging.",
            "- Interpretability: SVM models are less interpretable compared to decision trees or linear models."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- It is important to scale the features before fitting the SVM model, especially when using kernels such as RBF.",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/SVM/SVM.ipynb'],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/SVM/SVM.html"],
        'Documentation': ["https://scikit-learn.org/stable/modules/svm.html"],
        'Implementation Code': (
            "from sklearn.svm import SVC\n"
            "\n"
            "# Instantiate Support Vector Classifier\n"
            "model = SVC(kernel='linear')  # Use 'linear', 'poly', 'rbf', 'sigmoid' for different kernels\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }


# Dictionary containing model information
models_info['K Nearest Neighbors'] = {
        'Description': ["\n",
            "K-Nearest Neighbors (KNN) is a simple and effective supervised learning algorithm used for classification and regression tasks. In KNN, the prediction for a new data point is made by considering the majority class of its 'k' nearest neighbors in the feature space."
        ],
        'Type': ['Supervised Learning'],
        'SubType': ['Classification'],
        'Usage Scenarios': ["\n",
            "- Classification tasks with non-linear decision boundaries.",
            "- Regression tasks with non-linear relationships between features and target variable.",
            "- Outlier detection."
        ],
        'Advantages': ["\n",
            "- Simple and easy to implement.",
            "- No training phase: KNN stores all training data, making predictions fast.",
            "- Effective for small to medium-sized datasets.",
            "- Robust to noisy training data and outliers."
        ],
        'Limitations': ["\n",
            "- Computationally expensive during prediction, especially for large datasets.",
            "- Sensitive to the choice of distance metric and the value of 'k'.",
            "- Not suitable for high-dimensional data due to the curse of dimensionality."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- It is recommended to scale the features to ensure equal importance during distance calculations.",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/KNN/KNN.ipynb'],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/KNN/KNN.html"],
        'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"],
        'Implementation Code': (
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "\n"
            "# Instantiate KNN classifier\n"
            "model = KNeighborsClassifier(n_neighbors=5)  # Set the number of neighbors (k)\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }


# Dictionary containing model information
models_info['AdaBoost'] = {
        'Description': ["\n",
            "AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak learners to build a strong classifier. It iteratively trains weak models on the dataset, adjusting the weights of incorrectly classified instances to focus more on difficult-to-classify samples. The final prediction is made by weighted majority voting of all weak models."
        ],
        'Type': ['Supervised Learning'],
        'SubType': ['Classification | Ensemble Learning'],
        'Usage Scenarios': ["\n",
            "- Classification tasks with imbalance in class distribution.",
            "- Weak learners can be simple decision trees or other base classifiers.",
            "- AdaBoost can be effective in reducing bias and variance, leading to better generalization."
        ],
        'Advantages': ["\n",
            "- High accuracy and robustness in handling noisy data.",
            "- Can be used with various base classifiers.",
            "- Automatically handles feature selection and feature engineering.",
            "- Less prone to overfitting compared to individual weak learners."
        ],
        'Limitations': ["\n",
            "- Sensitive to noisy data and outliers, which can lead to overfitting.",
            "- Computationally expensive due to the iterative nature of training weak learners.",
            "- AdaBoost can be less effective when weak classifiers are too complex or when the data is highly skewed."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/AdaBoost/AdaBoost.ipynb'],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/AdaBoost/AdaBoost.html"],
        'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"],
        'Implementation Code': (
            "from sklearn.ensemble import AdaBoostClassifier\n"
            "\n"
            "# Instantiate AdaBoost classifier with base estimator (e.g., DecisionTreeClassifier)\n"
            "base_estimator = DecisionTreeClassifier(max_depth=1)  # Weak learner\n"
            "model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)  # 50 weak learners\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }

# Dictionary containing model information

models_info['Bagging'] = {
        'Description': ["\n",
            "Bagging (Bootstrap Aggregating) is an ensemble learning technique that aims to improve the stability and accuracy of machine learning algorithms. It works by training multiple base models (often decision trees) on different subsets of the training data, obtained by sampling with replacement (bootstrap samples). The final prediction is made by averaging (for regression) or voting (for classification) over predictions from all base models."
        ],
        'Type': ['Supervised Learning'],
        'SubType': ['Classification | Ensemble Learning'],
        'Usage Scenarios': ["\n",
            "- Classification and regression tasks with high variance or overfitting.",
            "- When the dataset is large and computational resources are limited.",
            "- Improving model performance by reducing variance and increasing stability."
        ],
        'Advantages': ["\n",
            "- Reduces overfitting by averaging or voting over multiple models.",
            "- Increases model stability and robustness.",
            "- Can be parallelized, making it efficient for large datasets.",
            "- Effective for improving performance of unstable models."
        ],
        'Limitations': ["\n",
            "- Limited interpretability: Bagging typically produces complex models that are difficult to interpret.",
            "- May not improve performance if the base models are already stable.",
            "- Requires additional computational resources for training multiple models."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/Bagging/Bagging.ipynb'],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/Bagging/Bagging.html"],
        'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html"],
        'Implementation Code': (
            "from sklearn.ensemble import BaggingClassifier\n"
            "from sklearn.tree import DecisionTreeClassifier\n"
            "\n"
            "# Instantiate base classifier (e.g., DecisionTreeClassifier)\n"
            "base_estimator = DecisionTreeClassifier()\n"
            "\n"
            "# Instantiate Bagging classifier\n"
            "model = BaggingClassifier(base_estimator=base_estimator, n_estimators=10)  # 10 base models\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }

# Dictionary containing model information


models_info['Gradient Boosting'] = {
        'Description': ["\n",
            "Gradient Boosting is an ensemble learning technique that builds a strong predictive model by combining the predictions of several weak learners, typically decision trees. Unlike traditional boosting methods, which focus on reducing the model's bias, Gradient Boosting minimizes the overall error by optimizing a loss function in the space of the model's predictions."
        ],
        'Type': ['Supervised Learning'],
        'SubType': ['Classification | Ensemble Learning'],
        'Usage Scenarios': ["\n",
            "- Regression and classification tasks where high predictive accuracy is desired.",
            "- When the dataset is relatively small and computational resources allow for training complex models.",
            "- Improving performance on structured data with heterogeneous features."
        ],
        'Advantages': ["\n",
            "- Produces highly accurate models by combining the strengths of weak learners.",
            "- Handles heterogeneous features and interactions between them effectively.",
            "- Less prone to overfitting compared to traditional boosting methods.",
            "- Robust to outliers in the data."
        ],
        'Limitations': ["\n",
            "- Gradient Boosting can be computationally expensive and memory-intensive, especially with large datasets.",
            "- Requires careful tuning of hyperparameters such as learning rate and tree depth.",
            "- May not perform well on high-dimensional sparse data."
        ],
        'Data Requirements': ["\n",
            "- Feature Matrix (X) with shape (n_samples, n_features).",
            "- Target Array (y) with shape (n_samples,).",
            "- X should be numeric or preprocessed to numeric form.",
            "- y can be numeric or string.",
            "- X and y should have consistent lengths."
        ],
        'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/GradientBoosting/GradientBoosting.ipynb'],
        'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/GradientBoosting/GradientBoosting.html"],
        'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"],
        'Implementation Code': (
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "\n"
            "# Instantiate Gradient Boosting classifier\n"
            "model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)  # Example hyperparameters\n"
            "\n"
            "# Train the model\n"
            "model.fit(X_train, y_train)\n"
            "\n"
            "# Predict labels for test data\n"
            "y_pred = model.predict(X_test)\n"
            "\n"
            "# Evaluate model performance\n"
            "accuracy = model.score(X_test, y_test)\n"
            "print('Accuracy:', accuracy)"
        )
    }

# Dictionary containing model information
models_info['XGBoost'] = {
    'Description': ["\n",
        "XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosting algorithms. It is highly efficient, scalable, and provides excellent performance. XGBoost builds a strong classifier by combining the predictions of multiple weak learners sequentially. It uses a gradient boosting framework and employs techniques like regularization to prevent overfitting and achieve high accuracy."
    ],
    'Type': ['Supervised Learning'],
    'SubType': ['Classification | Ranking | Ensemble Learning'],
    'Usage Scenarios': ["\n",
        "- Classification, regression, and ranking tasks in various domains.",
        "- Handling large datasets efficiently.",
        "- Achieving high predictive accuracy and interpretability."
    ],
    'Advantages': ["\n",
        "- Exceptional performance on a wide range of datasets.",
        "- Feature importance estimation for better interpretability.",
        "- Parallel and distributed computing support for scalability.",
        "- Built-in capabilities for handling missing values and regularization.",
        "- Flexible and customizable with various parameters for tuning."
    ],
    'Limitations': ["\n",
        "- Requires careful tuning of hyperparameters for optimal performance.",
        "- Longer training times compared to simpler models.",
        "- Not suitable for real-time applications due to computational complexity.",
        "- Memory-intensive, especially for large datasets."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Target Array (y) with shape (n_samples,).",
        "- X and y should have consistent lengths."
    ],
    'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Classification/XGBoost/XGBoost.ipynb'],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Classification/XGBoost/XGBoost.html"],
    'Documentation': ["https://xgboost.readthedocs.io/en/latest/index.html"],
    'Implementation Code': (
        "import xgboost as xgb\n"
        "\n"
        "# Instantiate XGBoost classifier\n"
        "model = xgb.XGBClassifier(n_estimators=100, max_depth=3)\n"
        "\n"
        "# Train the model\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "# Predict labels for test data\n"
        "y_pred = model.predict(X_test)\n"
        "\n"
        "# Evaluate model performance\n"
        "accuracy = accuracy_score(y_test, y_pred)\n"
        "print('Accuracy:', accuracy)"
    )
}

# Dictionary containing model information
models_info['KMeans'] = {
    'Description': ["\n",
        "KMeans is a popular unsupervised machine learning algorithm used for clustering tasks. It aims to partition a dataset into K distinct, non-overlapping clusters, where each data point belongs to the cluster with the nearest mean (centroid). KMeans iteratively assigns data points to the nearest centroid and updates the centroids until convergence. It is widely used for exploratory data analysis, customer segmentation, and image compression."
    ],
    'Type': ['Unsupervised Learning'],
    'SubType': ['Clustering'],
    'Usage Scenarios': ["\n",
        "- Identifying natural groupings within data.",
        "- Customer segmentation for targeted marketing.",
        "- Image compression and feature extraction."
    ],
    'Advantages': ["\n",
        "- Simple and easy to implement.",
        "- Scales well to large datasets.",
        "- Can handle high-dimensional data effectively.",
        "- Suitable for discovering spherical or elliptical clusters."
    ],
    'Limitations': ["\n",
        "- Requires the number of clusters (K) to be specified a priori.",
        "- Vulnerable to initialization sensitivity, leading to different solutions.",
        "- May converge to local optima depending on initial centroids.",
        "- Not suitable for clusters with non-linear boundaries or irregular shapes."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Preprocessing: Standardization or normalization of features is often recommended to ensure equal importance across dimensions. Outliers should also be handled appropriately as they can skew cluster assignments. Note that KMeans does not handle categorical variables directly, so encoding of categorical variables into numerical format might be necessary.",
        "- Determining the optimal number of clusters (K): This can be done using techniques such as the elbow method, silhouette score, or gap statistics."
    ],
    'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Clustering/Kmeans/KMeans.ipynb'],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Clustering/Kmeans/KMeans.html"],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"],
    'Implementation Code': (
        "from sklearn.cluster import KMeans\n"
        "\n"
        "# Instantiate KMeans clustering\n"
        "kmeans = KMeans(n_clusters=3, random_state=42)\n"
        "\n"
        "# Fit the model to the data\n"
        "clusters = kmeans.fit_predict(X)\n"
        "\n"
    )
}


# Dictionary containing model information
models_info['Hierarchical Clustering'] = {
    'Description': ["\n",
        "Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters. It starts with each data point as its cluster and then successively merges the nearest clusters into larger ones until all points are in a single cluster or until a stopping criterion is met. Hierarchical clustering produces a dendrogram, which can be cut at different levels to obtain a clustering of the data at different resolutions."
    ],
    'Type': ['Unsupervised Learning'],
    'SubType': ['Clustering'],
    'Usage Scenarios': ["\n",
        "- Identifying hierarchical structures within data.",
        "- Taxonomy construction.",
        "- Document clustering and gene expression analysis."
    ],
    'Advantages': ["\n",
        "- Does not require the number of clusters to be specified a priori.",
        "- Provides a hierarchy of clusters, allowing exploration at different levels of granularity.",
        "- Can handle datasets with non-linear relationships.",
        "- No sensitivity to initializations."
    ],
    'Limitations': ["\n",
        "- Computationally intensive for large datasets.",
        "- Produces fixed clusters once constructed; cannot directly accommodate new data points.",
        "- Choice of linkage method and distance metric can significantly affect the clustering result.",
        "- Interpretation of the dendrogram can be subjective."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Preprocessing: Standardization or normalization of features may be recommended.",
        "- Choice of linkage method: Different linkage methods (e.g., single, complete, average) may yield different results and should be chosen based on the dataset characteristics and clustering goals."
    ],
    'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/Clustering/HierarchicalClustering/HierarchicalClustering.ipynb'],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/Clustering/HierarchicalClustering/HierarchicalClustering.html"],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html"],
    'Implementation Code': (
        "from sklearn.cluster import AgglomerativeClustering\n"
        "\n"
        "# Instantiate Hierarchical Clustering\n"
        "hierarchical_clustering = AgglomerativeClustering(n_clusters=3)\n"
        "\n"
        "# Fit the model to the data\n"
        "hierarchical_clustering.fit(X)\n"
        "\n"
        "# Predict the cluster labels\n"
        "cluster_labels = hierarchical_clustering.labels_"
    )
}

# Dictionary containing model information
models_info['Principal Component Analysis'] = {
    'Description': ["\n",
        "Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving the most important information. It achieves this by identifying the principal components, which are the orthogonal vectors that capture the directions of maximum variance in the data. PCA is widely used for visualization, noise reduction, and feature extraction."
    ],
    'Type': ['Unsupervised Learning'],
    'SubType': ['Dimensionality Reduction'],
    'Usage Scenarios': ["\n",
        "- Dimensionality reduction: Reducing the number of features while retaining important information.",
        "- Visualization: Projecting high-dimensional data onto lower dimensions for visualization purposes.",
        "- Noise reduction: Filtering out noise from datasets.",
        "- Feature extraction: Identifying the most informative features for downstream tasks."
    ],
    'Advantages': ["\n",
        "- Reduces the dimensionality of data, making it computationally more efficient.",
        "- Retains most of the variance in the original data.",
        "- Allows for easy visualization of high-dimensional datasets.",
        "- Can improve the performance of machine learning algorithms by removing redundant or irrelevant features."
    ],
    'Limitations': ["\n",
        "- Assumes linear relationships among variables.",
        "- May not perform well on non-linear data distributions.",
        "- Interpretability of principal components may be challenging in high-dimensional spaces."
    ],
    'Data Requirements': ["\n",
        "- Feature Matrix (X) with shape (n_samples, n_features).",
        "- Preprocessing: Scaling of features is recommended to ensure features with larger scales do not dominate the analysis."
    ],
    'Example' : ['https://github.com/BAProjects/ML_Library/blob/main/DimensionalityReduction/PCA/PCA.ipynb'],
    'Render': ["https://raw.githubusercontent.com/BAProjects/ML_Library/main/DimensionalityReduction/PCA/PCA.html"],
    'Documentation': ["https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"],
    'Implementation Code': (
        "from sklearn.decomposition import PCA\n"
        "\n"
        "# Instantiate PCA with desired number of components\n"
        "pca = PCA(n_components=2)\n"
        "\n"
        "# Fit PCA to the data\n"
        "pca.fit(X)\n"
        "\n"
        "# Transform the data to its lower-dimensional representation\n"
        "X_pca = pca.transform(X)"
    )
}
