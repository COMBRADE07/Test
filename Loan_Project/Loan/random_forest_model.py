import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split

class Model:
    def model(self,custom_input):

        # Load the dataframe
        data = pd.read_csv("LoanApprovalPrediction.csv")
        print(data.columns)
        print(data.head(5))

        # Check for categorical variables in the dataset
        obj = (data.dtypes == 'object')
        print("Categorical variables:", len(list(obj[obj].index)))

        # Dropping Loan_ID column because there is no use for prediction
        data.drop(['Loan_ID'], axis=1, inplace=True)

        # Convert categorical variables to integers using LabelEncoder
        obj = (data.dtypes == 'object')
        object_cols = list(obj[obj].index)
        plt.figure(figsize=(18, 36))
        index = 1

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Apply label encoding to categorical columns
        for col in object_cols:
            data[col] = label_encoder.fit_transform(data[col])

        # Check the assigned integer values for each unique value
        for col in object_cols:
            class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            print(f"Mapping for {col}:")
            for class_label, encoded_value in class_mapping.items():
                print(f"{class_label}: {encoded_value}")
            print()

        # count categorical column
        obj = (data.dtypes == 'object')
        # print("Categorical variables:",len(list(obj[obj].index)))

        # heatmap to see correlation between variables
        plt.figure(figsize=(12, 6))
        sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f',
                    linewidths=2, annot=True)
        # plt.show()

        # checking missing values in dataset
        for col in data.columns:
            data[col] = data[col].fillna(data[col].mean())

        data.isna().sum()

        # extract dependent and independent variable in dataset
        X = data.drop(['Loan_Status'], axis=1)
        Y = data['Loan_Status']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        from sklearn.ensemble import RandomForestClassifier

        random_forest_classifier = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)

        # train model using random forest
        random_forest_classifier.fit(X, Y)

        print("X test values")
        print(len(X_test))

        # Convert the custom input to a DataFrame
        custom_input_df = pd.DataFrame(custom_input)

        # Preprocess the custom input
        for col in custom_input_df.select_dtypes(include='object'):
            custom_input_df[col] = label_encoder.transform(custom_input_df[col])

        p = random_forest_classifier.predict(custom_input_df)
        return p
