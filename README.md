# Machine Learning Model (Student Marks Prediction)

## 1. Importing Libraries:
- **NumPy**: Used for numerical operations.
- **Pandas**: Used for data manipulation and analysis.
- **Matplotlib**: Used for plotting graphs and visualizations.

## 2. Loading and Exploring the Dataset:
- Load the dataset using Pandas.
- Use `info()` to get a summary of the dataset's structure (columns, data types, etc.).
- Use `describe()` to get basic statistics of numerical columns (mean, std, min, max, etc.).

## 3. Visualizing the Data:
- Create a scatter plot to visualize the relationship between study hours and student marks.
- Label the x-axis as "Student Study Hours" and the y-axis as "Student Marks."

## 4. Data Cleaning:
- **Checking for Missing Values**:
  - Use `isnull().sum()` to check for missing values in each column.
- **Handling Missing Data**:
  - Fill missing values with the mean of their respective columns to ensure no missing data remains.

## 5. Preparing Data for Machine Learning:
- **Splitting Features and Labels**:
  - The features (X) are the study hours, and the labels (y) are the student marks.

## 6. Train-Test Split:
- Split the data into training and testing sets (80% training, 20% testing) using `train_test_split`.

## 7. Training the Model (Linear Regression):
- **Choosing a Model**: Use Linear Regression for predicting student marks based on study hours.
- **Training the Model**: Fit the model using the training data (X_train and y_train).
- **Understanding the Coefficients**:
  - The slope (m) indicates the rate of change between study hours and marks.
  - The intercept (c) tells the starting value of the marks when study hours is 0.
- **Model's Training Process**: Linear Regression fits a straight line through the data points, representing the relationship between study hours and student marks.
- **Why Linear Regression?**: Works well when there's a linear relationship between input (study hours) and output (marks).

## 8. Model Parameters:
- The model outputs a slope and an intercept, which are used in the equation `y = mx + c`.

## 9. Making Predictions:
- The model predicts the marks for a given number of study hours and for the test dataset.

## 10. Comparing Predictions with Actual Data:
- Compare the actual student marks and predicted marks using a DataFrame.

## 11. Model Accuracy:
- Calculate the accuracy score of the model on the test data to evaluate how well it performed.

## 12. Saving the Trained Model:
- Save the trained model using `joblib.dump()` to a file named "Student_marks_predictor_model.pkl".

## 13. Loading the Saved Model:
- Load the saved model using `joblib.load()` for future predictions without retraining.

## 14. Making Predictions with the Loaded Model:
- After loading the model, predict the marks for a new input (e.g., predicting the marks for a student studying 7 hours/day).

---

# Flask Application to Predict Student Marks

## Overview of app.py:

- **Imports**: Import necessary libraries including Flask and joblib.
- **Flask Application**: Initialize the Flask application and load the pre-trained model.
- **DataFrame Initialization**: Create an empty DataFrame to store input and prediction data.

### Routes:
- **Home Route**: Define the home page of the app, rendering the HTML template for user input.
- **Prediction Route**: Define a route to handle predictions, validate input, and predict student marks based on the number of study hours provided.

### Key Components:
1. **Input Handling**: Collect input values (study hours) from the web form and convert them to integers.
2. **Validation**: Ensure valid input (study hours should be between 0 and 24).
3. **Prediction**: Use the model to predict marks based on study hours.
4. **Storing Results**: Store the input and predicted output in a CSV file for later use.
5. **Output Display**: Display the predicted marks or adjust the display message if predicted marks exceed 100%.

### Running the Application:
- Start the web application using Flask, making it accessible on all network interfaces at a specified port.

## Project Structure:
- **Root Folder (student_marks_predictor/)**: Main project directory.
- **Subdirectories**:
  - `static/`: Contains static files like CSS, JavaScript, and image files.
  - `templates/`: Contains HTML files (index.html and result.html) for rendering the web interface.
  - `documentation/`: Stores project-related documentation.
  - `venv/`: Virtual environment folder.
- **Main Files**:
  - `app.py`: Main Flask application file.
  - `smp_data_from_app.csv`: Stores input data and predicted values.
  - `Student_marks_predictor_model.pkl`: The saved machine learning model.
- **External Libraries & Virtual Environment**:
  - Includes external dependencies managed by `venv`.

## Suggestions for Improvement:
1. **Modularize Code**: Consider splitting code into smaller files (e.g., moving model loading logic to `model.py`).
2. **Version Control**: Use Git for project tracking.
3. **Documentation**: Keep an updated `README.md` and detailed project documentation.
