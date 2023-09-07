import streamlit as st
import pandas as pd
import random
import classificadores
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
from funcs import *
 
# Define the Streamlit app
def main():
    # Streamlit app title
    st.title("Comparação dos modelos de classificação")
 
    # Get user-defined number of app_names
    num_app_names = st.number_input("Digite um número válido de amostras para validação:", min_value=2, step=1)
 
    # Load the DataFrame using the custom function
    df = carrega_df("df1")
    df["sentiment"] = df["review_score"].apply(lambda x: 1 if x == 1 else 0)
 
    results = []
    # Create a results DataFrame to store model metrics
    results_df = pd.DataFrame(columns=["Model", "Accuracy", "Recall", "Precision", "F1 Score", "Cross Validation", "Standard Deviation"])
 
    # Loop through each model and calculate metrics
    model_functions = [classificadores.support_vector_machine, classificadores.naive_bayes, classificadores.k_nearest, classificadores.regressao_logistica]  # Replace with your actual model functions
    # Randomly select app_names based on user-defined count
    random_app_names = random.sample(df["app_name"].tolist(), num_app_names)
    
    for i, model_func in enumerate(model_functions):
        model_name = f"Model: {model_func.__name__}"
        st.subheader(f"Results for {model_name}")
 
        # Initialize a dictionary to store model metrics for this model
        model_metrics = {"Model": model_name}
        
        # Calculate metrics for each app_name and display them
        for app_name in random_app_names:
            df_filtered = df[df["app_name"] == app_name]
            
            # Call the model function to calculate metrics
            accuracy, recall, precision, f1_score, mean_accuracy, standard_deviation = model_func(app_name, df_filtered)
            model_metrics[app_name] = {"Accuracy":accuracy, "Recall":recall, "Precision":precision, "F1 Score":f1_score, "Cross Validation":mean_accuracy, "Standard Deviation":standard_deviation}
            
        # Add the metrics to the results DataFrame
        results.append(model_metrics)
    
    def extract_values(row):
        if isinstance(row, list):
            if len(row) == 3 and isinstance(row[2], dict):
                return row[0], row[1], row[2]["Accuracy"], row[2]["Recall"], row[2]["Precision"], row[2]["F1 Score"], row[2]["Cross Validation"], row[2]["Standard Deviation"]
        return None, None, None, None, None
        
    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
 
 
 
    # Melt the DataFrame to reshape it for the model comparison table
    melted_results = results_df.melt(id_vars=["Model"], var_name="Game", value_name="Metrics")
    
    # Determine the unique keys in the "Metrics" column
    unique_keys = list(set(key for metric_dict in melted_results["Metrics"] if isinstance(metric_dict, dict) for key in metric_dict.keys()))
    
    # Create columns for each unique key
    for key in unique_keys:
        melted_results[key] = melted_results["Metrics"].apply(lambda x: x[key] if isinstance(x, dict) else None)
    
    melted_results.drop(columns=["Metrics"], inplace=True)
 
    # Compare model results and determine the winning model
    st.subheader("Comparação dos modelos")
 
    # Calculate the average accuracy for each game
    melted_results["Average Accuracy"] = melted_results.groupby("Game")["Accuracy"].transform("mean")
    # Calculate the average recall for each game
    melted_results["Average recall"] = melted_results.groupby("Game")["Recall"].transform("mean")
    # Calculate the average f1-score for each game
    melted_results["Average f1-score"] = melted_results.groupby("Game")["F1 Score"].transform("mean")
    
    winning_model_1 = melted_results[melted_results["Average Accuracy"] == melted_results.groupby("Game")["Average Accuracy"].transform("max")]["Model"].values
    winning_model_2 = melted_results[melted_results["Average recall"] == melted_results.groupby("Game")["Average recall"].transform("max")]["Model"].values
    winning_model_3 = melted_results[melted_results["Average f1-score"] == melted_results.groupby("Game")["Average f1-score"].transform("max")]["Model"].values
 
 
 
    # Calculate the average metrics for each model
    average_metrics = melted_results.groupby('Model')[['Recall', 'Accuracy', 'F1 Score']].mean()
    best_recall_value = average_metrics['Recall'].max()
    best_accuracy_value = average_metrics['Accuracy'].max()
    best_f1_score_value = average_metrics['F1 Score'].max()
 
    melted_results.drop(columns=["Average Accuracy"], inplace=True)
    melted_results.drop(columns=["Average recall"], inplace=True)
    melted_results.drop(columns=["Average f1-score"], inplace=True)
           
    st.write(melted_results)
    st.write(f"O modelo com a maior acurácia:\n{winning_model_1[0]}")
    st.write(f"Com uma acurácia de:\n{best_accuracy_value:.4f}")
    st.write(f"O modelo com o melhor recall:\n{winning_model_2[0]}")
    st.write(f"Com um recall de:\n{best_recall_value:.4f}")
    st.write(f"O modelo com o melhor f1-score:\n{winning_model_3[0]}")
    st.write(f"Com um f1-score de:\n{best_f1_score_value:.4f}")
 
 
main()
