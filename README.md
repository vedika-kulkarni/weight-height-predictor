# Weight and Height Prediction 📊

A Machine Learning project that analyzes the relationship between a person's height and weight and predicts weight using different regression algorithms. The project compares multiple regression models to identify the best-performing model based on evaluation metrics.
<img width="1452" height="800" alt="image" src="https://github.com/user-attachments/assets/c5a538d4-ed07-4cb3-9505-607179340ea5" />

## ✨ Features

- **Data Analysis** — Explore dataset structure, statistical summaries, and missing values.
- **Data Visualization** — Visualize the relationship between height and weight using scatter plots.
- **Data Preprocessing** — Split the dataset into training and testing sets and apply StandardScaler.
- **Multiple Regression Models** — Implement and compare five regression algorithms.
- **Model Evaluation** — Evaluate models using Mean Squared Error (MSE) and R-squared (R²).
- **Model Comparison** — Compare actual and predicted values through tables and visualizations.
- **Weight Prediction** — Predict weight using a given height.

## 🤖 Machine Learning Algorithms

The project implements the following regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regression
5. Random Forest Regression

## 📂 Dataset

The project uses the `weight-height.csv` dataset.

| Column | Description |
|---|---|
| Height | Person's height |
| Weight | Person's weight |

- **Input Feature:** Height
- **Target Variable:** Weight

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## 📈 Model Evaluation

The regression models are evaluated using:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values.
- **R-squared (R²):** Measures how well the model explains variations in the target variable.

The model with the highest R² score is selected for comparison as the best-performing model.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook

### Installation

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd YOUR_REPOSITORY_NAME
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `weight-height.csv` is present in the project directory.

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `weigth_height_prediction.ipynb` and run the cells.

## 📁 Project Structure

```text
Weight-Height-Prediction/
│
├── weigth_height_prediction.ipynb
├── weight-height.csv
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- Develop an interactive Streamlit prediction interface.
- Add more regression algorithms for comparison.
- Improve model performance through hyperparameter tuning.
- Deploy the prediction application online.

## 👩‍💻 Author

**Vedika Kulkarni**

GitHub: [vedika-kulkarni](https://github.com/vedika-kulkarni)
