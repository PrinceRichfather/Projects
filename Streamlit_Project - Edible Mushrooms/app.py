import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms poisonous or edible? 🍄")
    st.sidebar.markdown("Are your mushrooms poisonous or edible? 🍄")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('E:\Coursera Streamlit Mushrooms project\mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df['type']
        X = df.drop('type', axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    class_name = ['edible', 'poisonous']

    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_name)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()

        if "Preicision-Recall Curve" in metrics_list:
            st.subheader('Preicision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()



    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox('Classifier',
    ('Support Vector Machine (SVM)', 'Logistic Regression', 'Random Forest Classifier'))


    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Preicision-Recall Curve"))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_name).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_name).round(2) )
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider('Maximum number of iterations', 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Preicision-Recall Curve"))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_name).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_name).round(2) )
            plot_metrics(metrics)

    if classifier == 'Random Forest Classifier':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input('The maximum depth of a tree', 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio('Bootstrap samples when building trees', ("True", 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Preicision-Recall Curve"))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Random Forest Results')
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_name).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_name).round(2) )
            plot_metrics(metrics)



    if st.sidebar.checkbox('Show Raw Data', False):
        st.subheader('Mushrooms Dataset (Classification)')
        st.markdown("Dataset been label encoded using sklearn's LabelEncoder")
        st.write(df)







if __name__ == '__main__':
    main()
