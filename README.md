# MLBM_TabData
 
 Welcome to TabAnalyzer!

 To analyze your data with our tool, you simply need to upload your data file on the online application page:

    - in the "Data Analysis" Tab you can then explore your data (e.g. check response variable distribution),
      study plots of your data and get a general overview via the Profile Report

    - in the "Machine Learning" Tab you can then choose the settings for your machine learning model

        - Select Target Variable
        - Select Train Test Settings
            - Simple Split => is faster, good for a quick overview
            - Grid Search with CV => takes usually much longer to compute the results
                                     but in return you get the optimal hyperparameters for your model 
                                     and usually better and more robust results with your best model

        - Select Model Settings
            - select whether you want to perform classification or regression
            - depending on your choice, you can select between various classification or regression algorithms

        - Select Parameter Settings
            - if the "Grid Search with CV" option is active, you can here decide whether you want to perform 
              the grid search using default parameter or if you want to enter the values for your grid manually

        - Select Evaluation Metrics
            - depending on your analysis type (classification or regression), you can here select various         
              evaluation metrics for your model evaluation

    - if you hit the button "Train models", the model is applied to your data 
        - you will then get the evaluation metrics, the prediction output, result visualizations and an option to 
          download your trained model

 
