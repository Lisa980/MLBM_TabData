# import statements
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier 
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from io import BytesIO
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# -------------------- functions ------------------------
# function that reads in a file and returns dataframe based on its extension
def return_df(file):
    
    name = file.name

    # get the extension of the file
    extension = name.split(".")[-1]  

    # read data into dataframe adapted to the respective file type
    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep ="\t")
    elif extension == "json":
        df = pd.read_json(file)
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "xml":
        df = pd.read_xml(file)

    return df


# function to upload files
def file_uploader():

    # create streamlit file uploader
    f = st.file_uploader("Please upload the dataset", type = ["csv", "tsv", "json", "xlsx", "xml"])

    # if file uploader is used, read file into df 
    if f:
        df = return_df(f)
        st.success("File uploaded successfully")
        return df
    return None


# function that creates boxplots for all numerical features
def create_boxplots(df):

    # set figsize and select numerical features
    plt.figure(figsize=(12, 8))
    num_features = df.select_dtypes(include = ["float64", "int64"]).columns
    
    # if numerical features exist, create boxplot
    if len(num_features) > 0:
        df_melted = pd.melt(df, value_vars = num_features)
        sns.boxplot(x = "variable", y = "value", data = df_melted)
        plt.xticks(rotation = 90)
        plt.title('Boxplots of Numerical Features')
    else:
        st.write("No numerical features to plot.")

    plt.tight_layout()
    return plt


# function to create profile report
@st.cache_resource
def generate_profile_report(df):
    return ProfileReport(df, title = "Data Profile Report")


# function to create regression model
def regression_get_model(algorithm):
    
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42)
    elif algorithm == "Support Vector Regressor":
        model = SVR()
    elif algorithm == "Gradient Boosting Regressor":
        return GradientBoostingRegressor()
    elif algorithm == "Ridge Regression":
        return Ridge()
    elif algorithm == "Lasso Regression":
        return Lasso()
    
    return model


# function to create classification model
def classification_get_model(algorithm):

    if algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=42)
    elif algorithm == "Support Vector Classifier":
        model = SVC()
    elif algorithm == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
    elif algorithm == "K-Nearest Neighbors":
        return KNeighborsClassifier()
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    
    return model


# function for evaluation of classification models
def classification_eval(selected_metrics):

    # get selected evaulation metrics
    selected_metric_results = {}

    if "Accuracy" in selected_metrics:
        selected_metric_results["Accuracy"] = accuracy_score(y_test, y_pred)

    if "F1" in selected_metrics:
        selected_metric_results["F1"] = f1_score(y_test, y_pred, average = "weighted")  

    if "Precision" in selected_metrics:
        selected_metric_results["Precision"] = precision_score(y_test, y_pred, average = "weighted")

    if "Recall" in selected_metrics:
        selected_metric_results["Recall"] = recall_score(y_test, y_pred, average = "weighted")

    if "ROC/AUC" in selected_metrics:
        selected_metric_results["ROC/AUC"] = roc_auc_score(y_test, y_prob, multi_class = "ovr")

    if "MCC" in selected_metrics:
        selected_metric_results["MCC"] = matthews_corrcoef(y_test, y_pred)

    return selected_metric_results


# function for evaluation of regression models
def regression_eval(selected_metrics):

    # get selected evaulation metrics
    selected_metric_results = {}

    if "MSE" in selected_metrics:
        mse = mean_squared_error(y_test, y_pred)
        selected_metric_results["MSE"] = mse

    if "R2" in selected_metrics:
        r2 = r2_score(y_test, y_pred)
        selected_metric_results["R2"] = r2

    if "MAE" in selected_metrics:
        mae = mean_absolute_error(y_test, y_pred)
        selected_metric_results["MAE"] = mae

    return selected_metric_results


# function for evaluation of classification models when using grid search
def classification_eval_grid_search(best_model, X, y, selected_metrics, kf_cv):

    selected_metric_results = {}

    # get cross-validation scores
    cv_results = cross_validate(best_model, X, y, cv = kf_cv, scoring = {
        "accuracy" : "accuracy",
        "f1" : "f1_weighted",
        "precision" : "precision_weighted",
        "recall" : "recall_weighted",
        "roc_auc" : "roc_auc"
    }, return_train_score = False)

    if "Accuracy" in selected_metrics:
        selected_metric_results["Accuracy"] = cv_results["test_accuracy"].mean()

    if "F1" in selected_metrics:
        selected_metric_results["F1"] = cv_results["test_f1"].mean()

    if "Precision" in selected_metrics:
        selected_metric_results["Precision"] = cv_results["test_precision"].mean()

    if "Recall" in selected_metrics:
        selected_metric_results["Recall"] = cv_results["test_recall"].mean()

    if "ROC/AUC" in selected_metrics:
        selected_metric_results["ROC/AUC"] = cv_results["test_roc_auc"].mean()

    return selected_metric_results


# function for evaluation of regression models when using grid search
def regression_eval_grid_search(best_model, X, y, selected_metrics, kf_cv):

    selected_metric_results = {}

    # get cross-validation scores
    cv_results = cross_validate(best_model, X, y, cv = kf_cv, scoring = {
        "neg_mean_squared_error" : "neg_mean_squared_error",
        "neg_mean_absolute_error" : "neg_mean_absolute_error",
        "r2" : "r2"
    }, return_train_score = False)

    # convert negative metrics to positive values
    mse_scores = -cv_results["test_neg_mean_squared_error"]
    mae_scores = -cv_results["test_neg_mean_absolute_error"]

    if "MSE" in selected_metrics:
        selected_metric_results["MSE"] = mse_scores.mean()

    if "MAE" in selected_metrics:
        selected_metric_results["MAE"] = mae_scores.mean()

    if "R2" in selected_metrics:
        selected_metric_results["R2"] = cv_results["test_r2"].mean()

    return selected_metric_results

# -------------------- data preprocessing ------------------------

# function that encodes categorical data
def encode_data(input):

    # handle if input is a series
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = input.to_frame()

    # handle non-numerical variables using OneHotEncoder
    non_num = df.select_dtypes(include = ["object", "category"])

    encoders = {}
    df_num = df.copy()

    # for all non-numerical columns
    for column in non_num.columns:

        # encode the non-numerical column
        encoder = OneHotEncoder(sparse_output = False, drop = None)
        num_encoded = encoder.fit_transform(df_num[[column]])
        num_encoded_df = pd.DataFrame(num_encoded, columns = encoder.get_feature_names_out([column]))

        # drop original (categotical) column
        df_num = df_num.drop(columns = [column]).reset_index(drop = True)

        # concatenate the encoded column
        df_num = pd.concat([df_num, num_encoded_df], axis = 1)

        # store encoder
        encoders[column] = encoder

    return df_num


# function to split into train and test data
def sep_train_test(df, target, test_set_size):

    # define X and y
    X = df.drop(columns = target)
    y = df[target]

    # perform split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size)

    return X_train, X_test, y_train, y_test


# preprocessing pipeline
def preprocess_data(input, target):

    # handle if input is a series
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = input.to_frame()

    # access numerical and categorical columns separately
    col_cat = df.select_dtypes(include = ["object", "category"]).columns.tolist()
    col_num = df.select_dtypes(include = ["number"]).columns.tolist()

    # set up preprocessing pipeline for categorical data
    pipeline_cat = Pipeline([
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown = "ignore"))
    ])
    
    # set up preprocessing pipeline for numerical data
    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("std_scaler", StandardScaler())
    ])

    # use Column Transformer to combine the two pipelines
    pipeline_combined = ColumnTransformer([
        ("categorical", pipeline_cat, col_cat),
        ("numerical", pipeline_num, col_num)
    ])

    return pipeline_combined, col_num, col_cat


# -------------------- main streamlit application ------------------------
# create title
st.title("Machine Learning Application for Tabular Data Analysis")

# create file uploader
df = file_uploader()

# create division into data overview and machine learning part
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Machine Learning", "Model Application"])


# --------------------- settings for the data analysis tab ---------------------------------
with tab1:
    if df is not None:

        #### expander to display datatable
        with st.expander("Data Table", expanded = True):

            # display data table
            st.subheader("Data Table")
            st.dataframe(df)

        #### expander to display encoded datatable
        with st.expander("Encoded Data Table", expanded = True):

            # display encoded datatable
            st.subheader("Data Table after Encoding")
            df_enc = encode_data(df)
            st.dataframe(df_enc)

            # enable download of encoded datatable
            st.download_button(
                label = "Download Encoded Table",
                data = df_enc.to_csv(index = False).encode('utf-8'),
                file_name = "encoded_dataset.csv",
                mime = "text/csv"
            )

        #### expander to display response variable distribution
        with st.expander("Check Response Variable Distribution", expanded = True):

            # select target variable
            target = st.selectbox("**Select Target Variable**", df.columns)

            # check distribution of your selected target variable
            if target:

                # create histogram for the target variable
                target_distr, ax = plt.subplots(figsize = (8, 5))
                sns.histplot(df[target], ax = ax)

                # define histogram settings
                ax.set_title(f'Distribution of the selected target variable {target}')
                ax.set_xlabel(target)
                ax.set_ylabel('Frequency')

                # display histogram in streamlit app
                st.pyplot(target_distr)

                # enable download
                target_buf = BytesIO()
                target_distr.savefig(target_buf, format = "png")
                target_buf.seek(0) 

                st.download_button(
                    label = "Download Target Variable Distribution",
                    data = target_buf,
                    file_name = "target_var_distr.png",
                    mime = "image/png"
            )

        #### expander to display boxplots
        with st.expander("Boxplots", expanded = True):

            # display boxplot
            st.subheader("Boxplots of Numerical Features")
            boxplots = create_boxplots(df)
            st.pyplot(boxplots)

            # enable download of the boxplot
            buf = BytesIO()
            boxplots.savefig(buf, format = "png")
            buf.seek(0) 

            st.download_button(
                label = "Download Boxplots",
                data = buf,
                file_name = "boxplots.png",
                mime = "image/png"
            )

        #### expander to display pairplots
        with st.expander("Pairplots", expanded = True):
            # create pairplots
            pairplot = sns.pairplot(df)
            st.pyplot(pairplot)

            # enable download of the pairplots
            pair_buf = BytesIO()
            pairplot.savefig(pair_buf, format = "png")
            pair_buf.seek(0) 

            st.download_button(
                label = "Download Pairplots",
                data = pair_buf,
                file_name = "pairplots.png",
                mime = "image/png"
            )

                    # enable download of the boxplot
            buf = BytesIO()
            boxplots.savefig(buf, format = "png")
            buf.seek(0) 

            st.download_button(
                label = "Download Boxplots",
                data = buf,
                file_name = "boxplots.png",
                mime = "image/png"
            )

        #### expander to display ydata profile report
        with st.expander("Profile Report", expanded = True):

            # display profile report
            st.subheader("Profile Report")

            pr = generate_profile_report(df)
            st_profile_report(pr)

            # enable download of the profile report
            if st.button("Download Profile Report"):

                # write profile report to HTML
                html_pr = pr.to_html()
                buffer = BytesIO()
                buffer.write(html_pr.encode("utf-8"))
                buffer.seek(0)

                # create download button functionality
                st.download_button(
                    label = "Download Profile Report",
                    data = buffer,
                    file_name = "profile_report.html",
                    mime = "text/html"
                )

    else:
        st.write("Please upload a dataset!")


# --------------------- settings for the machine learning tab ---------------------------------
with tab2:
    if df is not None:

        # header of the tab
        st.header("Machine Learning Settings")

        # select target variable
        target = st.selectbox("**Select Target Variable**", df.columns, key = 2)

        # create options for train test split
        with st.expander("**Train/Test Settings**", expanded = True):

            # create option for cross validation
            cross_val_option = st.radio("**Select mode for Train/Test split**", ["Simple Split (no CV, no Grid Search)", "Grid Search with CV"])

            # if simple split is selected
            if cross_val_option == "Simple Split (no CV, no Grid Search)":

                if 'test_set_size' not in st.session_state:
                    st.session_state.test_set_size = 0.3

                # enable the user to select the size of the test set
                test_set_size = st.slider("**Select size of the test set:**", min_value = 0.1, max_value = 0.5, value = 0.3, step = 0.05)

                st.session_state.test_set_size = test_set_size

                # train test split
                X_train, X_test, y_train, y_test = sep_train_test(df, target, test_set_size)

                # preprocess data
                data_processed, col_num, col_cat = preprocess_data(X_train, target)

            # if cross validation split is selected
            elif cross_val_option == "Grid Search with CV":

                # enable the user to select the number of folds for CV
                fold_number = st.slider("**Select number of CV folds:**", min_value = 1, max_value = 15, value = 5, step = 1)

                # split into features and target
                X = df.drop(columns = target)
                y = df[target]

                # create KFold cross validation object
                kf_cv = KFold(n_splits = fold_number, shuffle = True, random_state = 42)

                # preprocess data
                data_processed, col_num, col_cat = preprocess_data(X, target)

        # create options for model parameters
        with st.expander('**Model Settings**', expanded = True):

            # user chooses machine learning mode and model type
            task_type = st.radio("**Select Analysis Type**", ["Classification", "Regression"])
            
            if task_type == "Classification":
                algorithm = st.selectbox("**Select Classification Algorithm**",
                        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier", "Gradient Boosting Classifier", "K-Nearest Neighbors", "Naive Bayes"])
                if algorithm:  
                    model = classification_get_model(algorithm)

            elif task_type == "Regression":
                algorithm = st.selectbox("**Select Regression Algorithm**",
                    ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor", "Gradient Boosting Regressor", "Ridge Regression", "Lasso Regression"])
                if algorithm:  
                    model = regression_get_model(algorithm)

            # enable user to choose between a grid search with default and with manually entered values
            model_set = st.radio("**Parameter Settings (only important if Grid Search is active)**", ["Default Grid", "Manual Grid"])

            # for each algorithm, provide default grid search settings
            if model_set == "Default Grid":

                # default grid settings for classification algorithms
                if algorithm == "Logistic Regression":
                    param_grid = {'model__C': [0.1, 1, 10]}
                elif algorithm == "Random Forest Classifier":
                    param_grid = {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20, 30]} 
                elif algorithm == "Support Vector Classifier":
                    param_grid = {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
                elif algorithm == "Gradient Boosting Classifier":
                    param_grid = {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 1], "model__max_depth": [3, 5, 7]}
                elif algorithm == "K-Nearest Neighbors":
                    param_grid = {"model__n_neighbors": [3, 5, 7], "model__weights": ["uniform", "distance"], "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}
                elif algorithm == "Naive Bayes":
                    param_grid = {}  

                # default grid settings for regression algorithms
                elif algorithm == "Linear Regression":
                    param_grid = {}
                elif algorithm == "Random Forest Regressor":
                    param_grid = {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20, 30]}
                elif algorithm == "Support Vector Regressor":
                    param_grid = {'model__C': [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
                elif algorithm == "Gradient Boosting Regressor":
                    param_grid = {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.1, 1], "model__max_depth": [3, 5, 7]}
                elif algorithm == "Ridge Regression":
                    param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
                elif algorithm == "Lasso Regression":
                    param_grid = {"model__alpha": [0.1, 1.0, 10.0]}

            # enable user to set grid search values (displays default values that can be changed)
            elif model_set == "Manual Grid":

                if algorithm == "Logistic Regression":
                    c_values = st.text_input("C (regularization strength)", value = " 0.1, 1, 10")

                    # convert string input into list and define parameter grid
                    c_values_conv = [float(val.strip()) for val in c_values.split(',')]
                    param_grid = {'model__C': c_values_conv}

                elif algorithm == "Random Forest Classifier":
                    n_estimators = st.text_input("n_estimators", value =  "50, 100, 200")
                    max_depth = st.text_input("max_depth", value = "10, 20, 30")  

                    # convert string input into list and define parameter grid
                    n_estimators_conv = [int(val.strip()) for val in n_estimators.split(",")]
                    max_depth_conv = [int(val.strip()) for val in max_depth.split(",")]
                    param_grid = {"model__n_estimators": n_estimators_conv, "model__max_depth" : max_depth_conv} 

                elif algorithm == "Support Vector Classifier":
                    c_values = st.text_input("C (regularization strength)", value = " 0.1, 1, 10")
                    kernels = st.selectbox("Kernel type", ["linear", "poly", "rbf", "sigmoid"])

                    # convert string input into list and define parameter grid
                    c_values_conv = [float(val.strip()) for val in c_values.split(",")]
                    param_grid = {'model__C': c_values_conv, "model__kernel" : [kernels]}

                elif algorithm == "Gradient Boosting Classifier":
                    n_estimators = st.text_input("n_estimators", value = "50, 100, 200")
                    learning_rate = st.text_input("learning_rate", value = "0.01, 0.1, 1")
                    max_depth = st.text_input("max_depth", value = "3, 5, 7")

                    # convert string input into list and define parameter grid
                    n_estimators_conv = [int(val.strip()) for val in n_estimators.split(",")]
                    learning_rate_conv = [float(val.strip()) for val in learning_rate.split(",")]
                    max_depth_conv = [int(val.strip()) for val in max_depth.split(",")]
                    param_grid = {"model__n_estimators": n_estimators_conv, "model__learning_rate": learning_rate_conv, "model__max_depth": max_depth_conv} 

                elif algorithm == "K-Nearest Neighbors":
                    n_neighbors = st.text_input("n_neighbors", value="5, 10, 15")
                    weights = st.selectbox("weights", ["uniform", "distance"])
                    p = st.selectbox("p (power parameter for Minkowski distance)", [1, 2])

                    # convert string input into list and define parameter grid
                    n_neighbors_conv = [int(val.strip()) for val in n_neighbors.split(",")]
                    param_grid = {"model__n_neighbors": n_neighbors_conv, "model__weights": [weights], "model__p": [p]}

                elif algorithm == "Naive Bayes":
                    st.write("Naive Bayes does not have hyperparameters to tune")
                    param_grid = {}

                elif algorithm == "Linear Regression":
                    st.write("No hyperparameter for training available")
                    param_grid = {}

                elif algorithm == "Random Forest Regressor":
                    n_estimators = st.text_input("n_estimators", value =  "50, 100, 200")
                    max_depth = st.text_input("max_depth", value = "10, 20, 30") 

                    # convert string input into list and define parameter grid
                    n_estimators_conv = [int(val.strip()) for val in n_estimators.split(",")]
                    max_depth_conv = [int(val.strip()) for val in max_depth.split(",")]
                    param_grid = {"model__n_estimators": n_estimators_conv, "model__max_depth" : max_depth_conv}  

                elif algorithm == "Support Vector Regressor":
                    c_values = st.text_input("C (regularization strength)", value = "0.1, 1, 10")
                    kernels = st.selectbox("Kernel type", ["linear", "poly", "rbf", "sigmoid"])

                    # convert string input into list and define parameter grid
                    c_values_conv = [float(val.strip()) for val in c_values.split(",")]
                    param_grid = {"model__C": c_values_conv, "model__kernel" : [kernels]}

                elif algorithm == "Gradient Boosting Regressor":
                    n_estimators = st.text_input("n_estimators", value = "100, 200, 300")
                    learning_rate = st.text_input("learning_rate", value = "0.01, 0.1, 0.2")
                    max_depth = st.text_input("max_depth", value = "3, 5, 7")

                    # convert string input into list and define parameter grid
                    n_estimators_conv = [int(val.strip()) for val in n_estimators.split(",")]
                    learning_rate_conv = [float(val.strip()) for val in learning_rate.split(",")]
                    max_depth_conv = [int(val.strip()) for val in max_depth.split(",")]
                    param_grid = {"model__n_estimators": n_estimators_conv, "model__learning_rate": learning_rate_conv, "model__max_depth": max_depth_conv} 

                elif algorithm == "Ridge Regression":
                    alpha = st.text_input("alpha (regularization strength)", value = "0.1, 1.0, 10.0")

                    # convert string input into list and define parameter grid
                    alpha_conv = [float(val.strip()) for val in alpha.split(",")]
                    param_grid = {"model__alpha": alpha_conv}

                elif algorithm == "Lasso Regression":
                    alpha = st.text_input("alpha (regularization strength)", value = "0.1, 1.0, 10.0")

                    # convert string input into list and define parameter grid
                    alpha_conv = [float(val.strip()) for val in alpha.split(",")]
                    param_grid = {"model__alpha": alpha_conv}

        # create options for model evaluation metrics
        if task_type == "Classification":
            eval_metrics = ["Accuracy", "F1", "Precision", "Recall", "ROC/AUC", "MCC"]

        elif task_type == "Regression":
            eval_metrics = ["MSE", "R2", "MAE"]

        if 'selected_metrics' not in st.session_state:
            st.session_state.selected_metrics = eval_metrics[:1] 

        # create multiselect option for the user to choose their desired evaluation metrics
        selected_metrics = st.multiselect("Select Evaluation Metrics", eval_metrics, default = eval_metrics[0])

        st.session_state.selected_metrics = selected_metrics

        # if user presses button to train models
        if st.button("**Train Models**"):
            st.markdown("**Results:**")

            # different options depending on selected train/test split

            #### if simple split option is selected ####
            if cross_val_option == "Simple Split (no CV, no Grid Search)":
                if data_processed is not None:

                    # create pipeline that combines preprocessing and model step
                    simple_pipe = Pipeline(steps = [
                        ("preprocessing", data_processed),
                        ("model", model)
                        ]) 
                    
                    st.write(f"Data preprocessing...")

                    # fit model using the pipeline
                    st.write(f"Fitting the model: {algorithm}...")

                    simple_pipe.fit(X_train, y_train)

                    # make prediction
                    y_pred = simple_pipe.predict(X_test) 

                    st.write("### Model Performance")

                    if task_type == "Classification":

                        # handle that Support Vector Classidier has no "predict_proba"
                        if algorithm != "Support Vector Classifier":

                            # get y_prob for ROC/AUC
                            y_prob = simple_pipe.predict_proba(X_test)[:, 1] 

                        # get selected evaluation metrics
                        selected_metric_results = classification_eval(selected_metrics)

                        # show selected metric results in streamlit app
                        if selected_metric_results:
                            for metric, value in selected_metric_results.items():

                                # format the metric value
                                if metric in ["Accuracy", "ROC/AUC"]:
                                    formatted_value = f"{value:.2%}"  
                                else:
                                    formatted_value = f"{value:.4f}"  
                                
                                # customize display 
                                metric_html = f"**{metric}:** <span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>{formatted_value}</span>"

                                # display the metric result with custom styling
                                st.markdown(metric_html, unsafe_allow_html=True)

                        # expander for the prediction outcome
                        with st.expander("Prediction Outcome", expanded = True):

                            # create predictions dataframe
                            predictions_df = pd.DataFrame({'Y_test': y_test,'Y_pred': y_pred})

                            # display prediction outcome in streamlit app
                            st.write("### Predicted Values:")
                            st.dataframe(predictions_df.reset_index(drop = True))

                            # enable download
                            csv_pred = predictions_df.to_csv(index = False).encode("utf-8")

                            st.download_button(
                                label = "Download Predictions",
                                data = csv_pred,
                                file_name = "predictions.csv",
                                mime = "text/csv"
                            )

                        # handle that Support Vector Classidier has no "predict_proba"
                        if algorithm != "Support Vector Classifier":
                            
                            # expander for ROC AUC
                            with st.expander("ROC/AUC Curve", expanded = True):

                                # compute ROC AUC
                                fpr, tpr, _ = roc_curve(y_test, y_prob)
                                roc_auc = auc(fpr, tpr)

                                # plot ROC Curve
                                roc_fig, ax = plt.subplots()
                                ax.plot(fpr, tpr, color = "#92c3ab", lw = 2, label = f"ROC curve (area = {roc_auc:.2f})")
                                ax.plot([0, 1], [0, 1], color = "#516287", lw = 2, linestyle = "--")
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.0])
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.set_title("Receiver Operating Characteristic")
                                ax.legend(loc = "lower right")
                                st.pyplot(roc_fig)

                                # enable download of the ROC/AUC curve
                                roc_buf = BytesIO()
                                roc_fig.savefig(roc_buf, format = "png")
                                roc_buf.seek(0) 

                                st.download_button(
                                    label = "Download ROC/AUC Curve",
                                    data = roc_buf,
                                    file_name = "roc_auc.png",
                                    mime = "image/png"
                                )
                            
                            # expander for precision-recall curve
                            with st.expander("Precision-Recall Curve", expanded = True):

                                # get precision and recall
                                precision, recall, _ = precision_recall_curve(y_test, y_prob)

                                # get area under the curve
                                precision_recall_auc = auc(recall, precision)

                                # plot precision-recall curve
                                prec_rec, ax = plt.subplots()
                                ax.plot(recall, precision, color = "#92c3ab", lw = 2, label = f"Precision-Recall curve (area = {precision_recall_auc:.2f})")

                                # define plot settings
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel("Recall")
                                ax.set_ylabel("Precision")
                                ax.set_title("Precision-Recall Curve")
                                ax.legend(loc = "lower left")

                                # display plot in streamlit app
                                st.pyplot(prec_rec)

                                # enable download of the Precision-Recall curve
                                pr_buf = BytesIO()
                                prec_rec.savefig(pr_buf, format = "png")
                                pr_buf.seek(0)

                                st.download_button(
                                    label = "Download Precision-Recall Curve",
                                    data = pr_buf,
                                    file_name = "precision_recall_curve.png",
                                    mime = "image/png"
                                )                           
                            
                        # expander for classification report
                        with st.expander("Classification Report", expanded = True):
                            
                            # display classification report
                            class_names = simple_pipe['model'].classes_
                            class_names = [str(cls) for cls in class_names]

                            class_report = classification_report(y_test, y_pred, target_names = class_names)
                            st.subheader("Classification Report")
                            st.text(class_report)  

                            # write classification report to HTML
                            html_class_rep = f"<pre>{class_report}</pre>"

                            buffer_class = BytesIO()
                            buffer_class.write(html_class_rep.encode('utf-8'))
                            buffer_class.seek(0)

                            # create download button functionality
                            st.download_button(
                                label = "Download Classification Report",
                                data = buffer_class,
                                file_name = "classification_report.html",
                                mime = "text/html"
                            )

                        # display confusion matrix
                        with st.expander("Confusion Matrix", expanded = True):

                            # create confusion matrix
                            conf_matrix = confusion_matrix(y_test, y_pred)

                            st.write("Confusion Matrix")

                            # display confusion matrix in streamlit app via seaborn
                            conf_heat, ax = plt.subplots(figsize = (6, 4))  
                            sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax, xticklabels = ["0", "1"], yticklabels = ["0", "1"])

                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            ax.set_title("Confusion Matrix")

                            st.pyplot(conf_heat)

                            # enable download of the confusion matrix heatmap
                            conf_buf = BytesIO()
                            conf_heat.savefig(conf_buf, format = "png")
                            conf_buf.seek(0)

                            st.download_button(
                                label = "Download Confusion Matrix",
                                data = conf_buf,
                                file_name = "confusion_matrix.png",
                                mime = "image/png"
                            )         

                    elif task_type == "Regression":

                        # get selected evaulation metrics
                        selected_metric_results = regression_eval(selected_metrics)

                        # display metrics
                        for metric, value in selected_metric_results.items():

                            # format metrics
                            formatted_value = f"{value:.4f}"
                            
                            # customize display
                            metric_html = f"**{metric}:** <span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>{formatted_value}</span>"
                            
                            # display metrics in streamlit app
                            st.markdown(metric_html, unsafe_allow_html=True)

                        # expander for prediction output
                        with st.expander("Prediction Outcome", expanded = True):

                            # create predictions dataframe
                            predictions_df = pd.DataFrame({'Y_test': y_test,'Y_pred': y_pred})

                            # display prediction outcome in streamlit app
                            st.write("### Predicted Values:")
                            st.dataframe(predictions_df.reset_index(drop = True))

                            # enable download
                            csv_pred = predictions_df.to_csv(index = False).encode("utf-8")

                            st.download_button(
                                label = "Download Predictions",
                                data = csv_pred,
                                file_name = "predictions.csv",
                                mime = "text/csv"
                            )

                        # expander for scatter plot
                        with st.expander("Scatter Plot", expanded = True):

                            # create scatter plot of actual vs. predicted values
                            plt.figure(figsize = (8, 6))
                            plt.scatter(y_test, y_pred, alpha = 0.7, color = "#92c3ab")
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw = 2)

                            # set labels and title
                            plt.xlabel("Actual Values")
                            plt.ylabel("Predicted Values")
                            plt.title("Actual vs Predicted Values")

                            # display scatterplot in streamlit app
                            st.pyplot(plt)

                            # enable download of the scatter plot
                            scatter_buf = BytesIO()
                            plt.savefig(scatter_buf, format = "png")
                            scatter_buf.seek(0) 

                            st.download_button(
                                label = "Download Scatter Plot",
                                data = scatter_buf,
                                file_name = "scatter_plot.png",
                                mime = "image/png"
                            )

                    # enable download of the trained model
                    st.download_button("Download Trained Model", data = pickle.dumps(simple_pipe), file_name = "model.pkl")

            ##### train model using the cross-validation train/test split ####
            elif cross_val_option == "Grid Search with CV":
                
                # create pipeline that combines preprocessing and model step
                cv_pipe = Pipeline(steps = [
                    ("preprocessing", data_processed),
                    ("model", model)
                    ])

                # perform grid search
                st.write(f"Performing Grid Search with {fold_number}-fold cross-validation for {algorithm}...")

                grid_search_cv = GridSearchCV(cv_pipe, param_grid, cv = kf_cv, scoring = "accuracy" if task_type == "Classification" else "neg_mean_squared_error")
                grid_search_cv.fit(X, y)

                # best model and its parameters
                best_model = grid_search_cv.best_estimator_
                best_params = grid_search_cv.best_params_ 

                # get standard deviation of the CV (best model)
                best_index = grid_search_cv.best_index_
                cv_std = grid_search_cv.cv_results_['std_test_score'][best_index]

                # display best parameters
                st.markdown(f"### Best Parameters for {algorithm}:")
                formatted_params = [
                    f"**Best value for hyperparameter {key.replace('model__', '')}:** <span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>{value}</span>"
                    for key, value in best_params.items()
                ]
                formatted_params_str = "<br>".join(formatted_params)
                st.markdown(formatted_params_str, unsafe_allow_html = True)

                # display standard deviation of the CV
                st.markdown(f"##### Standard Deviation of the Best Model's Cross-Validation Score:")
                st.markdown(f"<span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>"
                            f"{cv_std:.4f}</span>", unsafe_allow_html = True)
                
                # get prediction (needed for scatter plot)
                y_pred = best_model.predict(X)

                # get model performance evaluation
                st.write("### Model Performance")

                if task_type == "Classification":

                    # cross-validation results
                    selected_metric_results = classification_eval_grid_search(best_model, X, y, selected_metrics, kf_cv)

                    # show selected metric results in Streamlit app
                    if selected_metric_results:
                        for metric, value in selected_metric_results.items():
                            # format the metric results
                            if metric in ["Accuracy", "ROC/AUC"]:
                                formatted_value = f"{value:.2%}"  # Display as percentage for Accuracy and ROC/AUC
                            else:
                                formatted_value = f"{value:.4f}"  # Display with four decimal places for others

                            metric_html = f"**{metric}:** <span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>{formatted_value}</span>"

                            # display the metric results in the streamlit app
                            st.markdown(metric_html, unsafe_allow_html = True)

                    # expander for the prediction outcome
                    with st.expander("Prediction Outcome", expanded = True):

                        # create predictions dataframe
                        predictions_df = pd.DataFrame({'Y_test': y,'Y_pred': y_pred})

                        # display prediction outcome in streamlit app
                        st.write("### Predicted Values:")
                        st.dataframe(predictions_df.reset_index(drop = True))

                        # enable download
                        csv_pred = predictions_df.to_csv(index = False).encode("utf-8")

                        st.download_button(
                            label = "Download Predictions",
                            data = csv_pred,
                            file_name = "predictions.csv",
                            mime = "text/csv"
                        )

                    # handle that Support Vector Classidier has no "predict_proba"
                    if algorithm != "Support Vector Classifier":

                        # get y_prob
                        y_prob = best_model.predict_proba(X)[:, 1]
                        
                        # expander for ROC AUC
                        with st.expander("ROC/AUC Curve", expanded = True):

                            # compute ROC AUC
                            fpr, tpr, _ = roc_curve(y, y_prob)
                            roc_auc = auc(fpr, tpr)

                            # plot ROC Curve
                            roc_fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, color = "#92c3ab", lw = 2, label = f"ROC curve (area = {roc_auc:.2f})")
                            ax.plot([0, 1], [0, 1], color = "#516287", lw = 2, linestyle = "--")
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.0])
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.set_title("Receiver Operating Characteristic")
                            ax.legend(loc = "lower right")
                            st.pyplot(roc_fig)

                            # enable download of the ROC/AUC curve
                            roc_buf = BytesIO()
                            roc_fig.savefig(roc_buf, format = "png")
                            roc_buf.seek(0) 

                            st.download_button(
                                label = "Download ROC/AUC Curve",
                                data = roc_buf,
                                file_name = "roc_auc.png",
                                mime = "image/png"
                            )
                        
                        # expander for precision-recall curve
                        with st.expander("Precision-Recall Curve", expanded = True):

                            # get precision and recall
                            precision, recall, _ = precision_recall_curve(y, y_prob)

                            # get area under the curve
                            precision_recall_auc = auc(recall, precision)

                            # plot precision-recall curve
                            prec_rec, ax = plt.subplots()
                            ax.plot(recall, precision, color = "#92c3ab", lw = 2, label = f"Precision-Recall curve (area = {precision_recall_auc:.2f})")

                            # define plot settings
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel("Recall")
                            ax.set_ylabel("Precision")
                            ax.set_title("Precision-Recall Curve")
                            ax.legend(loc = "lower left")

                            # display plot in streamlit app
                            st.pyplot(prec_rec)

                            # enable download of the Precision-Recall curve
                            pr_buf = BytesIO()
                            prec_rec.savefig(pr_buf, format = "png")
                            pr_buf.seek(0)

                            st.download_button(
                                label = "Download Precision-Recall Curve",
                                data = pr_buf,
                                file_name = "precision_recall_curve.png",
                                mime = "image/png"
                            )                           
                        
                    # expander for classification report
                    with st.expander("Classification Report", expanded = True):

                        # access class names
                        model = best_model.named_steps['model']
                        if hasattr(model, 'classes_'):
                            class_names = model.classes_

                        # handle if some ML model types have no attribute "classes"
                        else:
                            class_names = np.unique(y)

                        class_names = [str(cls) for cls in class_names]

                        # display classification report
                        class_report = classification_report(y, y_pred, target_names = class_names)
                        st.subheader("Classification Report")
                        st.text(class_report)  

                        # write classification report to HTML
                        html_class_rep = f"<pre>{class_report}</pre>"

                        buffer_class = BytesIO()
                        buffer_class.write(html_class_rep.encode('utf-8'))
                        buffer_class.seek(0)

                        # create download button functionality
                        st.download_button(
                            label = "Download Classification Report",
                            data = buffer_class,
                            file_name = "classification_report.html",
                            mime = "text/html"
                        )

                    # display confusion matrix
                    with st.expander("Confusion Matrix", expanded = True):

                        # create confusion matrix
                        conf_matrix = confusion_matrix(y, y_pred)

                        st.write("Confusion Matrix")

                        # display confusion matrix in streamlit app via seaborn
                        conf_heat, ax = plt.subplots(figsize = (6, 4))  
                        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax, xticklabels = ["0", "1"], yticklabels = ["0", "1"])

                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")

                        st.pyplot(conf_heat)

                        # enable download of the confusion matrix heatmap
                        conf_buf = BytesIO()
                        conf_heat.savefig(conf_buf, format = "png")
                        conf_buf.seek(0)

                        st.download_button(
                            label = "Download Confusion Matrix",
                            data = conf_buf,
                            file_name = "confusion_matrix.png",
                            mime = "image/png"
                        )         


                    st.download_button("Download Trained Model", data = pickle.dumps(best_model), file_name = "model.pkl")


                elif task_type == "Regression":

                    # cross-validation results
                    selected_metric_results = regression_eval_grid_search(best_model, X, y, selected_metrics, kf_cv)

                    # display metrics
                    for metric, value in selected_metric_results.items():

                        # format metrics
                        formatted_value = f"{value:.4f}"
                        
                        # customize display
                        metric_html = f"**{metric}:** <span style='background-color: #92c3ab; color: #155724; padding: 2px 4px; border-radius: 4px;'>{formatted_value}</span>"
                        
                        # display metrics in streamlit app
                        st.markdown(metric_html, unsafe_allow_html=True)

                    # expander for prediction output
                    with st.expander("Prediction Outcome", expanded = True):

                        # create predictions dataframe
                        predictions_df = pd.DataFrame({'Y_test': y,'Y_pred': y_pred})

                        # display prediction outcome in streamlit app
                        st.write("### Predicted Values:")
                        st.dataframe(predictions_df.reset_index(drop = True))

                        # enable download
                        csv_pred = predictions_df.to_csv(index = False).encode("utf-8")

                        st.download_button(
                            label = "Download Predictions",
                            data = csv_pred,
                            file_name = "predictions.csv",
                            mime = "text/csv"
                        )

                    # expander for scatter plot
                    with st.expander("Scatter Plot", expanded = True):

                        # create scatter plot of actual vs. predicted values
                        plt.figure(figsize = (8, 6))
                        plt.scatter(y, y_pred, alpha = 0.7, color = "#92c3ab")
                        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw = 2)

                        # set labels and title
                        plt.xlabel("Actual Values")
                        plt.ylabel("Predicted Values")
                        plt.title("Actual vs Predicted Values")

                        # display scatterplot in streamlit app
                        st.pyplot(plt)

                        # enable download of the scatter plot
                        scatter_buf = BytesIO()
                        plt.savefig(scatter_buf, format = "png")
                        scatter_buf.seek(0) 

                        st.download_button(
                            label = "Download Scatter Plot",
                            data = scatter_buf,
                            file_name = "scatter_plot.png",
                            mime = "image/png"
                        )

                    # enable download of the trained model
                    st.download_button("Download Trained Model", data = pickle.dumps(best_model), file_name = "model.pkl")  
        else:
            st.write("Please upload a dataset!")


# --------------------- settings for the machine learning tab ---------------------------------

# Note: this is only an idea how one could continue working on this app;
#       unfortuately I did not have enough time to continue here

with tab3:
    st.header("Model Application")
    st.markdown("**Make predictions for a new dataset of your choice using your trained model from the ML tool:**")

    # upload for new datasets
    second_upload = st.file_uploader("Please upload your new dataset here:", type = ["csv", "tsv", "json", "xlsx", "xml"], key = "new_data_uploader")

    if second_upload:
        new_data = return_df(second_upload)
        st.success("File uploaded successfully")

    # upload for already trained models
    model_upload = st.file_uploader("If you already have a trained model saved to a file, you can upload it here:", type = ["pkl"], key ="model_uploader")

    if model_upload:
        model = model_upload
        st.success("Model uploaded successfully")