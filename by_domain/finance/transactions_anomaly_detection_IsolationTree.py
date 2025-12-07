
# based on https://medium.com/@smruti.po1106/anomaly-detection-in-transactions-d7e10e90f01f

import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def load_dataset() -> pd.DataFrame:
    # Load the dataset from a CSV file
    df = pd.read_csv("data/transaction_anomalies_dataset.csv")
    return df

def show_data_distribution(data: pd.DataFrame, column: str,nbins: int = 20):
    # Distribution of Transaction Amount
    fig_amount = px.histogram(data, x=column,nbins=nbins,title=f'Distribution of {column}')
    fig_amount.show()

def show_correlation_of_all_columns(data: pd.DataFrame):
    # Correlation Heatmap
    correlation_matrix = data.corr()
    print (correlation_matrix)
    fig_corr_heatmap = px.imshow(correlation_matrix,
                                title='Correlation Heatmap')
    fig_corr_heatmap.show()

def show_normal_vs_anomalous(data: pd.DataFrame):
    # Scatter plot of Transaction Amount with anomalies highlighted
    fig_anomalies = px.scatter(data, x='Transaction_Amount', y='Average_Transaction_Amount',
                           color='is_anomaly', title='Anomalies in Transaction Amount')
    fig_anomalies.update_traces(marker=dict(size=12), 
                            selector=dict(mode='markers', marker_size=1))
    fig_anomalies.show()

def flag_anomalies(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Flag anomalies in the specified column based on a simple statistical method.
    Anomalies are defined as values that are more than 2 standard deviations from the mean.

    we can take the transaction amount and flag every transaction that the is greater then the thresolg (mean_deviation + 2*std_deviation) as an anomaly


    """
    # Calculate mean and standard deviation of column
    mean_deviation = data[column].mean()
    std_deviation = data[column].std()
    # Define the anomaly threshold (2 standard deviations from the mean)
    threshold = mean_deviation + 2 * std_deviation
    data['is_anomaly'] = data[column] > threshold
    return data

def train_isolation_forest_model(X_train: pd.DataFrame) -> IsolationForest:
    """
    Train an Isolation Forest model on the training data.
    """
    X = data[['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']]
    y = data ['is_anomaly'] 
    print (y)
    X_data, X_test, y_data,  y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(X_data)
    test_isolation_forest_model(model, X_test, y_test)
    return model

def test_isolation_forest_model(model: IsolationForest, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Test the Isolation Forest model on the test data and print classification report.
    """
    y_pred = model.predict(X_test)
    # convert  -1/1 to 0/1 0=normal , 1=anomaly
    y_pred = [1 if x==-1 else 0 for x in y_pred]
    print (y_pred)
    print (classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

if __name__=='__main__':
    print ("Transactions")
    data = load_dataset()
    print (data.columns)
    print (data.head())   
#    show_data_distribution(data, 'Transaction_Amount')
#    show_correlation_of_all_columns(data)
    # create a new column 'is_anomaly' to flag anomalies where 'Transaction_Amount' >2*std_deviation from the mean deviation
    data=flag_anomalies(data, 'Transaction_Amount')
    print(data[data['is_anomaly'] == True])
#    show_normal_vs_anomalous(data)

    model = train_isolation_forest_model(data)

    # Now we can use the model

    input = pd.DataFrame([{
        'Transaction_Amount': 5000,
        'Average_Transaction_Amount': 200,
        'Frequency_of_Transactions': 5}]
        )
    
    print(input)

    prediction=model.predict(input)
    print (prediction)  # -1 for anomaly, 1 for normal


    






