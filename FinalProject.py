import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import category_encoders as ce
import matplotlib.pyplot as plt


# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    demonstrateHelpers(trainDF)
    
    '''
    Uses one-hot encoding 
    Uncomment Preprocess 1 in transformData, comment out Preprocess 2 and 3 in transformData, comment out
    preprocess2Exp and preprocess3Exp before running
    '''
    #preprocess1Exp(trainDF, testDF)
    
    '''
    The preprocess combines some attributes together
    Uncomment Preprocess 2 in transformData, comment out Preprocess 1 and 3 in transformData, comment out
    preprocess1Exp and preprocess3Exp before running
    '''
    #preprocess2Exp(trainDF, testDF)
    
    '''
    Uses binary encoding + combines some attributes together
    Uncomment Preprocess 3 in transformData, comment out Preprocess 1 and 2 in transformData, comment out
    preprocess1Exp and preprocess2Exp before running
    '''
    preprocess3Exp(trainDF, testDF)

# =============================================================================
'''
Standardize and normalize funtions
'''
def standardize(df, listOfColumns):
    df.loc[:, listOfColumns] = ((df.loc[:, listOfColumns]-df.loc[:, listOfColumns].mean())/(df.loc[:, listOfColumns].std()))

def normalize (df, listOfColumns):
    df.loc[:, listOfColumns] = ((df.loc[:, listOfColumns]-df.loc[:, listOfColumns].min())/(df.loc[:, listOfColumns].max()-df.loc[:, listOfColumns].min()))
    
# ===============================================================================
'''
Preprocess by one-hot encoding. To standardize/normalize, uncomment the standardize/normalize line 
'''
def preprocess(targetDF, sourceDF):
    # Variables that we need to preprocess
    '''We try to add more attributes'''
    '''label = ["MSZoning", "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", 
             "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", 
             "Electrical", "KitchenQual", "Functional", "GarageType", "PavedDrive", 
             "PoolQC", "SaleCondition", "GarageQual", "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature",
             "Street", "Alley"]'''
    
    label = ["MSZoning", "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", 
             "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", 
             "Electrical", "KitchenQual", "Functional", "GarageType", "PavedDrive", 
             "PoolQC"]
    
    '''One-hot encoding for all the attributes in the "label" variable'''
    targetDF = pd.get_dummies(targetDF, columns=label)      

    newLabel = targetDF.columns.tolist()
    
    return targetDF, newLabel

# ===============================================================================
'''
Preprocess 2. To standardize/normalize, uncomment the standardize/normalize line
'''
def preprocess2(targetDF, sourceDF):
    '''We try to add more attributes'''
    '''label = ["MSZoning", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", 
            "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", 
            "OverallCond", "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "BsmtQual", 
            "BsmtCond", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", 
            "Functional", "GarageType", "PavedDrive", "PoolQC", "SaleCondition", "GarageQual", 
            "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature", "Street", "Alley"]'''
    
    label = ["MSZoning", "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", 
             "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", 
             "Electrical", "KitchenQual", "Functional", "GarageType", "PavedDrive", 
             "PoolQC"]
    targetDF = pd.get_dummies(targetDF, columns=label) 
    '''Create a new attribute buildingAge that contains the age of the building'''
    
    targetDF.loc[:, "buildingAge"] = 2021 - sourceDF.loc[:, "YearBuilt"] 
    
    '''Create a new attribute remodAge that contains the age of the building since remodeling'''
    targetDF.loc[:, "remodAge"] = 2021 - sourceDF.loc[:, "YearRemodAdd"]
    
    '''Replace NA values with 0'''
    targetDF.loc[:, "TotalBsmtSF"] = sourceDF.loc[:, "TotalBsmtSF"].fillna(0)
    sourceDF.loc[:, "TotalBsmtSF"] = sourceDF.loc[:, "TotalBsmtSF"].fillna(0)
    targetDF.loc[:, "BsmtUnfSF"] = sourceDF.loc[:, "BsmtUnfSF"].fillna(0)
    sourceDF.loc[:, "BsmtUnfSF"] = sourceDF.loc[:, "BsmtUnfSF"].fillna(0)
    '''Create a new attribute percentBsmtFinished that contains the percentage of finished basement '''
    targetDF.loc[:, "percentBsmtFinished"] = sourceDF.apply(lambda row: (row.loc["TotalBsmtSF"] - row.loc["BsmtUnfSF"]) / 
                                                            row.loc["TotalBsmtSF"] if row.loc["TotalBsmtSF"] > 0 
                                                            else row.loc["TotalBsmtSF"], axis = 1)
    
    '''Create a new attribute total1stAnd2ndSF that contains the total square feet of 1st and 2nd floors'''
    targetDF.loc[:, "total1stAnd2ndSF"] = sourceDF.loc[:, "1stFlrSF"] + sourceDF.loc[:, "2ndFlrSF"]

    '''Replace NA values with 0'''
    targetDF.loc[:, "BsmtFullBath"] = sourceDF.loc[:, "BsmtFullBath"].fillna(0)
    sourceDF.loc[:, "BsmtFullBath"] = sourceDF.loc[:, "BsmtFullBath"].fillna(0)
    
    '''Create a new attribute totalFullBath that contains the total square feet of full bath'''
    targetDF.loc[:, "totalFullBath"] = sourceDF.loc[:, "BsmtFullBath"] + sourceDF.loc[:, "FullBath"]
    
    '''Replace NA values with 0'''
    targetDF.loc[:, "BsmtHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"].fillna(0)
    sourceDF.loc[:, "BsmtHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"].fillna(0)
    '''Create a new attribute totalHalfBath that contains the total square feet of half bath'''
    targetDF.loc[:, "totalHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"] + sourceDF.loc[:, "HalfBath"]
    
    '''Replace NA values with 2021'''
    targetDF.loc[:, "GarageYrBlt"] = sourceDF.loc[:, "GarageYrBlt"].fillna(2021)
    sourceDF.loc[:, "GarageYrBlt"] = sourceDF.loc[:, "GarageYrBlt"].fillna(2021)
    
    '''Create a new attribute garageAge that contains the age of the garage'''
    targetDF.loc[:, "garageAge"] = 2021 - sourceDF.loc[:, "GarageYrBlt"]
    
    '''Create a new attribute totalFullBath that contains the total square feet of totalPorchSF'''
    targetDF.loc[:, "totalPorchSF"] = (sourceDF.loc[:, "OpenPorchSF"] + sourceDF.loc[:, "EnclosedPorch"] + 
                                       sourceDF.loc[:, "3SsnPorch"] + sourceDF.loc[:, "ScreenPorch"])
    
    '''Drop the columns that were used to create combined attributes'''
    targetDF = targetDF.drop(columns = ["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "BsmtUnfSF", "1stFlrSF", "2ndFlrSF", 
                             "BsmtFullBath", "FullBath", "BsmtHalfBath", "HalfBath", "GarageYrBlt", "OpenPorchSF",
                             "EnclosedPorch", "3SsnPorch", "ScreenPorch"])
    
    '''Attributes need to be standardized/normalized'''
    stdList = ["LotArea", "TotRmsAbvGrd", "totalPorchSF", "garageAge", "totalHalfBath", 
               "totalFullBath", "total1stAnd2ndSF", "percentBsmtFinished", "remodAge", "buildingAge"]
    
    #standardize(targetDF, stdList)
    #normalize(targetDF, stdList)
    newLabel = targetDF.columns.tolist()
    return targetDF, newLabel 

# ===============================================================================
# BEGIN: from https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/
# EXPLANATION: Built-in binary encoding
# import category_encoders as ce
# encoder= ce.BinaryEncoder(cols = label, return_df = True)
# trainDF = encoder.fit_transform(sourceDF)
# END: from https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/

'''
!!!!!!!!!NOTE!!!!!!!!!!
Run 
!pip install category_encoders 
in the console to install the library for binary encoding
To standardize/normalize, uncomment the standardize/normalize line
'''
def preprocess3(targetDF, sourceDF):
    '''We try to add more attributes'''
    '''label = ["MSZoning", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", 
            "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", 
            "OverallCond", "Exterior1st", "Exterior2nd", "ExterQual", "ExterCond", "BsmtQual", 
            "BsmtCond", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", 
            "Functional", "GarageType", "PavedDrive", "PoolQC", "SaleCondition", "GarageQual", 
            "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature", "Street", "Alley"]'''
    
    label = ["MSZoning", "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", 
             "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", 
             "Electrical", "KitchenQual", "Functional", "GarageType", "PavedDrive", 
             "PoolQC"]
    encoder= ce.BinaryEncoder(cols = label, return_df = True)
    targetDF = encoder.fit_transform(sourceDF)
    
    '''Create a new attribute buildingAge that contains the age of the building'''    
    targetDF.loc[:, "buildingAge"] = 2021 - sourceDF.loc[:, "YearBuilt"] 
    
    '''Create a new attribute remodAge that contains the age of the building since remodeling'''
    targetDF.loc[:, "remodAge"] = 2021 - sourceDF.loc[:, "YearRemodAdd"]
    
    '''Replace NA values with 0'''
    targetDF.loc[:, "TotalBsmtSF"] = sourceDF.loc[:, "TotalBsmtSF"].fillna(0)
    sourceDF.loc[:, "TotalBsmtSF"] = sourceDF.loc[:, "TotalBsmtSF"].fillna(0)
    targetDF.loc[:, "BsmtUnfSF"] = sourceDF.loc[:, "BsmtUnfSF"].fillna(0)
    sourceDF.loc[:, "BsmtUnfSF"] = sourceDF.loc[:, "BsmtUnfSF"].fillna(0)
    '''Create a new attribute percentBsmtFinished that contains the percentage of finished basement '''
    targetDF.loc[:, "percentBsmtFinished"] = sourceDF.apply(lambda row: (row.loc["TotalBsmtSF"] - row.loc["BsmtUnfSF"]) / 
                                                            row.loc["TotalBsmtSF"] if row.loc["TotalBsmtSF"] > 0 
                                                            else row.loc["TotalBsmtSF"], axis = 1)
    
    '''Create a new attribute total1stAnd2ndSF that contains the total square feet of 1st and 2nd floors'''
    targetDF.loc[:, "total1stAnd2ndSF"] = sourceDF.loc[:, "1stFlrSF"] + sourceDF.loc[:, "2ndFlrSF"]

    '''Replace NA values with 0'''
    targetDF.loc[:, "BsmtFullBath"] = sourceDF.loc[:, "BsmtFullBath"].fillna(0)
    sourceDF.loc[:, "BsmtFullBath"] = sourceDF.loc[:, "BsmtFullBath"].fillna(0)
    
    '''Create a new attribute totalFullBath that contains the total square feet of full bath'''
    targetDF.loc[:, "totalFullBath"] = sourceDF.loc[:, "BsmtFullBath"] + sourceDF.loc[:, "FullBath"]
    
    '''Replace NA values with 0'''
    targetDF.loc[:, "BsmtHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"].fillna(0)
    sourceDF.loc[:, "BsmtHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"].fillna(0)
    '''Create a new attribute totalHalfBath that contains the total square feet of half bath'''
    targetDF.loc[:, "totalHalfBath"] = sourceDF.loc[:, "BsmtHalfBath"] + sourceDF.loc[:, "HalfBath"]
    
    '''Replace NA values with 2021'''
    targetDF.loc[:, "GarageYrBlt"] = sourceDF.loc[:, "GarageYrBlt"].fillna(2021)
    sourceDF.loc[:, "GarageYrBlt"] = sourceDF.loc[:, "GarageYrBlt"].fillna(2021)
    
    '''Create a new attribute garageAge that contains the age of the garage'''
    targetDF.loc[:, "garageAge"] = 2021 - sourceDF.loc[:, "GarageYrBlt"]
    
    '''Create a new attribute totalFullBath that contains the total square feet of totalPorchSF'''
    targetDF.loc[:, "totalPorchSF"] = (sourceDF.loc[:, "OpenPorchSF"] + sourceDF.loc[:, "EnclosedPorch"] + 
                                       sourceDF.loc[:, "3SsnPorch"] + sourceDF.loc[:, "ScreenPorch"])
    
    '''Drop the columns that were used to create combined attributes'''
    targetDF = targetDF.drop(columns = ["YearBuilt", "YearRemodAdd", "TotalBsmtSF", "BsmtUnfSF", "1stFlrSF", "2ndFlrSF", 
                             "BsmtFullBath", "FullBath", "BsmtHalfBath", "HalfBath", "GarageYrBlt", "OpenPorchSF",
                             "EnclosedPorch", "3SsnPorch", "ScreenPorch"])
    
    '''Attributes need to be standardized/normalized'''
    stdList = ["LotArea", "TotRmsAbvGrd", "totalPorchSF", "garageAge", "totalHalfBath", 
               "totalFullBath", "total1stAnd2ndSF", "percentBsmtFinished", "remodAge", "buildingAge"]
    #standardize(targetDF, stdList)
    #normalize(targetDF, stdList)
    
    newLabel = targetDF.columns.tolist()
    return targetDF, newLabel

# ============================================================================
# Data cleaning - conversion, normalization
'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF, predictors):
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    
    '''The original trainInput and testInput row length'''
    trainInputLen = trainInput.shape[0]
    testInputLen = testInput.shape[0]

    #countTrain = 0
    #countTest = 0
    
    '''Add what's missing from trainInput to testInput and vice versa'''
    for col in predictors:
        for row in range(len(trainInput.loc[:, col])):
            if testInput.loc[:, col].isin([trainInput.loc[row, col]]).any() == False:
                testInput.append(trainInput.loc[row, :])
                #countTest += 1
            if (row < len(testInput.loc[:, col])) and (trainInput.loc[:, col].isin([testInput.loc[row, col]]).any() == False):
                trainInput.append(testInput.loc[row, :])
                #countTrain += 1
            #print("countTest: " + str(countTest))
            #print("countTrain: " + str(countTrain))

    '''
    Calling Preprocess on the trainInput and testInput
    '''
    trainInputTemp = trainInput.copy()
    
    '''Preprocess 1'''
    '''trainInput, predictors = preprocess(trainInput, trainInputTemp)
    testInput, predictors = preprocess(testInput, trainInputTemp)'''
    
    
    
    '''Preprocess 2'''
    '''trainInput, predictors = preprocess2(trainInput, trainInputTemp)
    testInput, predictors = preprocess2(testInput, trainInputTemp)'''
    
    '''Preprocess 3'''
    trainInput, predictors = preprocess3(trainInput, trainInputTemp)
    testInput, predictors = preprocess3(testInput, trainInputTemp)
    
    
    '''Remove the added rows to get the original data'''
    while trainInput.shape[0] != trainInputLen:
        trainInput = trainInput.iloc[:-1, :]
    while testInput.shape[0] != testInputLen:
        testInput = testInput.iloc[:-1, :]
    
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
   
# =============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])
    
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResultsLinearRegression.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=14, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    return cvMeanScore

# ===============================================================================
'''
Use the kNN algorithm to predict
'''
def kNNTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = KNeighborsRegressor(n_neighbors = 8)
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    predictions = alg.predict(testInput.loc[:, predictors])
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })
    
    accuracies = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=14, scoring='r2')
    print("kNN Algorithm accuracy: ", np.mean(accuracies))
    # Prepare CSV
    submission.to_csv('data/testResultsKNN.csv', index=False)
    return np.mean(accuracies)
    
# ===============================================================================
# BEGIN: from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# EXPLANATION: Built-in Gradient Boosting Regressor
# from sklearn.ensemble import GradientBoostingRegressor
# reg = GradientBoostingRegressor(random_state=0)
# reg.fit(X_train, y_train)
# reg.predict(X_test[1:2])
# END: from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

'''
Use Gradient Boosting to predict
'''
def gradientBoostingTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = GradientBoostingRegressor()
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    predictions = alg.predict(testInput.loc[:, predictors])
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })
    accuracies = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=14, scoring='r2')
    print("Gradient Boosting Algorithm accuracy: ", np.mean(accuracies))
    # Prepare CSV
    submission.to_csv('data/testResultsGradBoost.csv', index=False)
    return np.mean(accuracies)

# =============================================================================
'''
Experiments with preprocess 1
'''
def preprocess1Exp(trainDF, testDF):
    '''We try to add more attributes'''
    '''label = ["MSZoning", "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", 
             "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", 
             "Electrical", '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 
             'Functional', 'GarageType', 'PavedDrive', 'OpenPorchSF', "KitchenQual", 
             "PoolQC", "SaleCondition", "GarageQual", "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature",
             "Street", "Alley"]'''
    
    label = ['MSZoning', 'LotArea', 'LotShape', 'LandContour', 'Utilities', 
             'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
             'HouseStyle', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 
             'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Heating', 'HeatingQC', 'CentralAir', 
             'Electrical', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 
             'Functional', 'GarageType', 'PavedDrive', 'OpenPorchSF', 'PoolQC']

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF, label)
    print("===============Linear Regression===============")
    r2ScoreLR = doExperiment(trainInput, trainOutput, predictors)
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============kNN===============")
    r2ScoreKNN = kNNTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============Gradient Boosting===============")
    r2ScoreGB = gradientBoostingTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    '''Create a bar graph to compare the prediction models' scores'''
    r2Score = [r2ScoreLR, r2ScoreKNN, r2ScoreGB]
    r2Graph(r2Score)

# =============================================================================
'''
Experiments with preprocess 2
'''
def preprocess2Exp(trainDF, testDF):
    '''We try to add more attributes'''
    '''label = ["MSZoning", 'LotArea', "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", "ExterQual", 
             "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", "Electrical", 
             '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Functional', 'GarageType', 
             'PavedDrive', 'OpenPorchSF', "KitchenQual", "PoolQC", "SaleCondition", "GarageQual", 
             "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature", "Street", "Alley", "YearBuilt", 
             'YearRemodAdd', "TotalBsmtSF", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", 
             "EnclosedPorch", "3SsnPorch", "ScreenPorch"]'''
    
    label = ['MSZoning', 'LotArea', 'LotShape', 'LandContour', 'Utilities', 
             'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
             'HouseStyle', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 
             'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Heating', 'HeatingQC', 'CentralAir', 
             'Electrical', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 
             'Functional', 'GarageType', 'PavedDrive', 'OpenPorchSF', 'PoolQC', "YearBuilt", 
             "TotalBsmtSF", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "EnclosedPorch", 
             "3SsnPorch", "ScreenPorch"]
    
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF, label)
    print("===============Linear Regression===============")
    r2ScoreLR = doExperiment(trainInput, trainOutput, predictors)
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============kNN===============")
    r2ScoreKNN = kNNTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============Gradient Boosting===============")
    r2ScoreGB = gradientBoostingTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    '''Create a bar graph to compare the prediction models' scores'''
    r2Score = [r2ScoreLR, r2ScoreKNN, r2ScoreGB]
    r2Graph(r2Score)

# =============================================================================
'''
Experiments with preprocess 3
'''
def preprocess3Exp(trainDF, testDF):
    '''We try to add more attributes'''
    '''label = ["MSZoning", 'LotArea', "LotShape", "LandContour", "Utilities", 
             "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
             "HouseStyle", "OverallQual", "OverallCond", "Exterior1st", "Exterior2nd", "ExterQual", 
             "ExterCond", "BsmtQual", "BsmtCond", "Heating", "HeatingQC", "CentralAir", "Electrical", 
             '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Functional', 'GarageType', 
             'PavedDrive', 'OpenPorchSF', "KitchenQual", "PoolQC", "SaleCondition", "GarageQual", 
             "FireplaceQu", "RoofStyle", "SaleType", "MiscFeature", "Street", "Alley", "YearBuilt", 
             'YearRemodAdd', "TotalBsmtSF", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", 
             "EnclosedPorch", "3SsnPorch", "ScreenPorch"]'''
    
    label = ['MSZoning', 'LotArea', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
             'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 
             'OverallCond', 'YearRemodAdd', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 
             'BsmtQual', 'BsmtCond', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', 
             '2ndFlrSF', 'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'GarageType', 
             'PavedDrive', 'OpenPorchSF', 'PoolQC', "YearBuilt", "TotalBsmtSF", "BsmtUnfSF", "BsmtFullBath", 
             "BsmtHalfBath", "GarageYrBlt", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF, label)
    print("===============Linear Regression===============")
    r2ScoreLR = doExperiment(trainInput, trainOutput, predictors)
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============kNN===============")
    r2ScoreKNN = kNNTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    print("===============Gradient Boosting===============")
    r2ScoreGB = gradientBoostingTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    '''Create a bar graph to compare the prediction models' scores'''
    r2Score = [r2ScoreLR, r2ScoreKNN, r2ScoreGB]
    r2Graph(r2Score)
    
# ===============================================================================
# BEGIN: from https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# EXPLANATION: Built-in function to draw bar plot 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# langs = ['C', 'C++', 'Java', 'Python', 'PHP']
# students = [23,17,35,29,12]
# ax.bar(langs,students)
# plt.show()
# def r2Graph(r2Score): # ADDED: Make it works as a function that takes in a list of r2 scores
# END: from https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm

'''A bar graph to graph r2 score'''
def r2Graph(r2Score):
    algs = ["Linear Regression", "kNN", "Gradient Boosting"]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(algs, r2Score)
    plt.show()

# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

