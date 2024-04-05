import streamlit as st

def models_list():

    col1, col2 = st.columns((1.5,1))
    
    with col1:
        st.subheader('Supervised Learning',divider='green')

        col11, col12, col13 = st.columns(3)

        with col11:
            st.write("Classification")
            st.write('''
                    - Decision Trees
                    - Logistic Regression
                    - Naive Bayes Classifier
                    - K-Nearest neighbors
                    - Support Vector Machines
                     ''')
        
        with col12:
            st.write("Regression")
            st.write('''
                    - Linear Regression
                    - Ridge Regeression
                    - Lasso Regression
                    - Neural Network Regressor
                     ''')

        with col13:
            st.write("Ensemble")
            st.write("""
                    - Random Forrest
                    - Gradient Boosting
                    - AdaBoost
                    - Bagging
                    - XGBoost
                     """)

    with col2:
        st.subheader("Unsupervised Learning",divider='blue')
        col21, col22 = st.columns(2)

        with col21:
            st.write('Clustering')
            st.write('''
                    - K-means clustering
                    - hierarchical clustering
                    ''')

        with col22:
            st.write("Dimensionality Reduction")
            st.write('''
                    - Principal Component Analysis
                    ''')

# Define homepage content
def homepage_content():
    st.header("Machine Learning Cookbook")
    
    st.write("Browse through our list of machine learning models and find implementation examples:")
    models_list()
    # Add links to other models or sections here
    st.write("---")
    st.write("Have feedback for us? Reach out to us at changing.signals@gmail.com")

    st.caption(":red[Disclaimer:]")
    st.caption('''
This app serves as a quick reference guide for machine learning models and practical examples. It aims to assist users in exploring and selecting the available models for their projects. While efforts have been made to ensure accuracy, the content is for reference only and may not cover all scenarios. Users are encouraged to use their discretion and verify information independently.

By using this app, you acknowledge its purpose and understand that it does not guarantee specific outcomes.
             ''')
