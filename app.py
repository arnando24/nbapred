import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Define the Streamlit app
def main():
    # Title of the app
    st.title('NBA MVP Prediction by Arnando Harlianto') 

    # Create sidebar with tabs
    app_mode = st.sidebar.selectbox("Choose the Algorithm Model", ["Decision Tree", "Linear Regression", "Support Vector Regression"])

    # Display content based on selected tab
    if app_mode == "Decision Tree":
        # Show data upload widget
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            sorted_data2223 = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
                
            # Check if required columns are present
            required_columns = ['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']
            if not set(required_columns).issubset(sorted_data2223.columns):
                st.warning("The selected dataset does not contain all required columns.")
                return

            # Data splitting
            if st.button("Start Predict & Visualize"):
                X = sorted_data2223[['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']]
                y = sorted_data2223['fantasy_points']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model_dt = DecisionTreeRegressor()
                model_dt.fit(X_train, y_train)

                # Make predictions
                y_pred = model_dt.predict(X_test)

                # Display results
                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                results_sorted = results.sort_values(by='Predicted', ascending=False)

                # Visualize results
                fig, ax = plt.subplots()
                ax.scatter(results_sorted['Actual'], results_sorted['Predicted'])
                ax.set_xlabel('Actual Fantasy Points')
                ax.set_ylabel('Predicted Fantasy Points')
                ax.set_title('Actual vs Predicted Fantasy Points (Sorted)')
                st.pyplot(fig)

                # Make predictions on entire dataset
                y_final_test = model_dt.predict(X)
                new_x = pd.concat([X, y], axis=1)
                new_x["predicted_value"] = y_final_test
                final_df = new_x[["fantasy_points", "predicted_value"]]
                st.write("Predictions on Entire Dataset:")
                st.dataframe(final_df, width=None)

                # Calculate R^2 score
                accuracy_percentage_dt = r2_score(y_test, y_pred) * 100
                st.write("Prediction Accuracy Percentage (R^2 score): {:.2f}%".format(accuracy_percentage_dt))

    elif app_mode == "Linear Regression":
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            sorted_data2223 = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
           
            # Check if required columns are present
            required_columns = ['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']
            if not set(required_columns).issubset(sorted_data2223.columns):
                st.warning("The selected dataset does not contain all required columns.")
                return

            # Data splitting
            if st.button("Start Predict & Visualize"):
                X = sorted_data2223[['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']]
                y = sorted_data2223['fantasy_points']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)

                # Make predictions
                y_pred = model_lr.predict(X_test)

                # Display results
                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                results_sorted = results.sort_values(by='Predicted', ascending=False)

                # Visualize results
                fig, ax = plt.subplots()
                ax.scatter(results_sorted['Actual'], results_sorted['Predicted'])
                ax.set_xlabel('Actual Fantasy Points')
                ax.set_ylabel('Predicted Fantasy Points')
                ax.set_title('Actual vs Predicted Fantasy Points (Sorted)')
                st.pyplot(fig)

                # Make predictions on entire dataset
                y_final_test = model_lr.predict(X)
                new_x = pd.concat([X, y], axis=1)
                new_x["predicted_value"] = y_final_test
                final_df = new_x[["fantasy_points", "predicted_value"]]
                st.write("Predictions on Entire Dataset:")
                st.dataframe(final_df, width=None)

                # Calculate R^2 score
                accuracy_percentage_lr = r2_score(y_test, y_pred) * 100
                st.write("Prediction Accuracy Percentage (R^2 score): {:.2f}%".format(accuracy_percentage_lr))

    elif app_mode == "Support Vector Regression":
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            sorted_data2223 = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
            
            # Check if required columns are present
            required_columns = ['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']
            if not set(required_columns).issubset(sorted_data2223.columns):
                st.warning("The selected dataset does not contain all required columns.")
                return

            # Data splitting
            if st.button("Start Predict & Visualize"):
                X = sorted_data2223[['age', 'games_played', 'wins', 'loses', 'minutes_played', 'points', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', '3_point_made', '3_point_attempted', '3_point_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'personal_fouls', 'double_doubles', 'triple_doubles']]
                y = sorted_data2223['fantasy_points']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model_svr = SVR()
                model_svr.fit(X_train, y_train)

                # Make predictions
                y_pred = model_svr.predict(X_test)

                # Display results
                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                results_sorted = results.sort_values(by='Predicted', ascending=False)

                # Visualize results
                fig, ax = plt.subplots()
                ax.scatter(results_sorted['Actual'], results_sorted['Predicted'])
                ax.set_xlabel('Actual Fantasy Points')
                ax.set_ylabel('Predicted Fantasy Points')
                ax.set_title('Actual vs Predicted Fantasy Points (Sorted)')
                st.pyplot(fig)

                # Make predictions on entire dataset
                y_final_test = model_svr.predict(X)
                new_x = pd.concat([X, y], axis=1)
                new_x["predicted_value"] = y_final_test
                final_df = new_x[["fantasy_points", "predicted_value"]]
                st.write("Predictions on Entire Dataset:")
                st.dataframe(final_df, width=None)

                # Calculate R^2 score
                accuracy_percentage_svr = r2_score(y_test, y_pred) * 100
                st.write("Prediction Accuracy Percentage (R^2 score): {:.2f}%".format(accuracy_percentage_svr))

if __name__ == "__main__":
    main()
