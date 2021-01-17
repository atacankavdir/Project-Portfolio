import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def create_class_dicts():
    #List all csv files, read,open and assign them to dataframes. The output is a dictionary with 11 dataframes as values
    import glob
    path = r'D:\Pythonworkspace\Data Science Edu\Applied ML\ML-PROJECT' # use your path
    all_files = glob.glob(path + "/*.csv")

    csv_dict={}
    for filename in all_files:
        csv_name = filename[filename.find('T\\')+2:filename.find('.')]
        csv_dict[csv_name] = pd.read_csv(filename, index_col=None).fillna(0)


    #Create a seperate dictionary for each class (A,B,C) and a new dataframe for our target 
    GPANew = csv_dict['GPANew']
    class_A_list = list(csv_dict.keys())[:3]
    class_B_list = list(csv_dict.keys())[3:6]
    class_C_list = list(csv_dict.keys())[6:10]

    #creates new dicts with same type of classes
    for i in ['A', 'B', 'C']:
        globals()['class_'+i+'_dict']= {}
        for year in locals()['class_'+i+'_list']:
            globals()['class_'+i+'_dict'][year] = csv_dict[year]
        print('class_'+i+'_dict is ready')
    return class_A_list, class_B_list, class_C_list, class_A_dict, class_B_dict, class_C_dict,GPANew



def final_score_calculator(class_dict, class_name = 'A'):
    '''
    Function calculates final scores of student according to furmulas mentioned in project document
    Inputs:
            class_dict: Dictionary of a class which contains related dataframes of available years
            class_name: Identifies the subject class which will be aggregated
    Output:
            aggregated_dict_X: Returns an aggregated column and grades of students for X class with respect to years
    '''
    if class_name == 'A':
        aggregated_dict_A = {}
        for key, value in class_dict.items():
            FORMULA_PART_MT = (value.loc[:,'MT1'] +value.loc[:,'MT2'] +value.loc[:,'MT3'])
            FORMULA_PART_HW = (value.loc[:,'HW1'] + value.loc[:,'HW2'] + value.loc[:,'HW3'])
            if key == 'A2013New':
                value['FINAL_SCORE'] = 0.3*FORMULA_PART_MT  + (0.1/3)*FORMULA_PART_HW 
            elif key == 'A2014New':
                value['FINAL_SCORE'] = 0.3*FORMULA_PART_MT  + (1/3)* (value.loc[:,'HW1']/7 + value.loc[:,'HW2']/6 + value.loc[:,'HW3']/6)
            elif key == 'A2015New':
                value['FINAL_SCORE'] = 0.3*FORMULA_PART_MT  + (1/7)*FORMULA_PART_HW 
            aggregated_dict_A[key] = value[['FINAL_SCORE', 'Grade']]
        return aggregated_dict_A
    elif class_name == 'B':
        aggregated_dict_B = {}
        for key, value in class_dict.items():
            FORMULA_PART_MT_PRJ = 0.25*(value.loc[:,'MT1'] + value.loc[:,'MT2']) + 0.3*value.loc[:,'MT3'] + 0.15*value.loc[:,'PRJ'] 
            if key == 'B2013New':
                value['FINAL_SCORE'] = FORMULA_PART_MT_PRJ  + value.loc[:,'Q1']
            elif key == 'B2014New':
                value['FINAL_SCORE'] = FORMULA_PART_MT_PRJ  + 0.05*value.loc[:,'Q1']
            elif key == 'B2015New':
                value['FINAL_SCORE'] = FORMULA_PART_MT_PRJ  + 0.05*value.loc[:,'Q1']
            aggregated_dict_B[key] = value[['FINAL_SCORE', 'Grade']]
        return aggregated_dict_B
    elif class_name == 'C':
        aggregated_dict_C = {}
        for key, value in class_dict.items(): 
            if key == 'C2013New':
                value['FINAL_SCORE'] = 0.25*(value.loc[:,'MT1'] +value.loc[:,'MT2']) + 0.3*value.loc[:,'MT3'] + 0.15*value.loc[:,'PRJ'] +value.loc[:,'Q1']
            elif key == 'C2014New':
                value['FINAL_SCORE'] = 0.25*(value.loc[:,'MT1']) + 0.3*(value.loc[:,'MT2']+value.loc[:,'MT3']) + 0.1*value.loc[:,'PRJ'] +value.loc[:,'Q1']
            elif key == 'C2015New':
                value['FINAL_SCORE'] = 0.25*(value.loc[:,'MT1']) + 0.3*(value.loc[:,'MT2']+value.loc[:,'MT3']) + 0.1*value.loc[:,'PRJ'] +value.loc[:,'Q1']
            elif key == 'C2018New':
                value['FINAL_SCORE'] = 0.3*(value.loc[:,'MT1'] + value.loc[:,'MT2'])+0.375*value.loc[:,'MT3']+0.025*value.loc[:,'Q1']    
            aggregated_dict_C[key] = value[['FINAL_SCORE', 'Grade']]
        return aggregated_dict_C


def grade_categorizer(aggregated_dict):
    '''
    Function simply transform Grade column in to binary form: 
        1: Passed
        0: Failed
    '''
    categorized_dict = aggregated_dict.copy()
    for value in aggregated_dict.values():
        value.loc[:,'Grade'] = np.where(value.loc[:,'Grade'] == 'F', 0, 1)
    return categorized_dict     
    
    
def scatter_plot_creater(x_axis, y_axis, x_axis_label, y_axis_label):
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    ax.scatter(x_axis,y_axis)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title('Class Dispersion')
    plt.show()
    
    
def my_decision_tree(TRX,TRY,TSX,TSY):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(np.array(TRX).reshape(-1, 1),TRY)
    print('                   ')
    print('Class prediction: ')
    print('                   ')
    #Predict the response for test dataset
    y_pred = clf.predict(np.array(TSX).reshape(-1, 1))
    print(confusion_matrix(TSY,y_pred))
    print('                   ')
    print ("Accuracy Score:", accuracy_score(TSY, y_pred))
    
    
#define a cross validation calculator function
def lets_cross_validate(classifier, feature_df, target, cv, scoring):
    cv_score = cross_val_score(classifier, feature_df, target, cv=cv, scoring=scoring)
    print('Accuracy Score of each fold')
    print(cv_score)
    print("Mean Accuracy Score - " + '\033[4m'+ str(cv_score.mean()) + '\033[0m')
    return cv_score, cv_score.mean()