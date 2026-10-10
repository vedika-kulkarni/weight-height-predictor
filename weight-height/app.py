import os
import sys
import pickle
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Model & Scaler Helper
# -------------------------------------------------------------
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
CSV_FILE = "weight-height.csv"
DOWNLOADS_CSV = os.path.expanduser(r"~\Downloads\weight-height.csv")

def get_scaler_and_model():
    """
    Loads or reconstructs the scaler and regression model.
    Handles corrupted/string pickle artifacts gracefully.
    """
    scaler = None
    model = None

    # 1. Load or recreate Scaler
    if os.path.exists(SCALER_FILE):
        try:
            with open(SCALER_FILE, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None

    if scaler is None or not hasattr(scaler, "transform"):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fitted values from training dataset (mean ~66.368, std ~3.848)
        scaler.mean_ = np.array([66.367560])
        scaler.scale_ = np.array([3.847528])
        scaler.var_ = np.array([14.803472])
        scaler.n_features_in_ = 1

    # 2. Load or recreate Model
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                loaded = pickle.load(f)
                if hasattr(loaded, "predict"):
                    model = loaded
        except Exception:
            model = None

    if model is None:
        from sklearn.linear_model import LinearRegression
        
        # Check if dataset is available to fit
        csv_path = CSV_FILE if os.path.exists(CSV_FILE) else (DOWNLOADS_CSV if os.path.exists(DOWNLOADS_CSV) else None)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                X = df[['Height']]
                y = df['Weight']
                X_scaled = scaler.transform(X)
                model = LinearRegression()
                model.fit(X_scaled, y)
                # Save the trained model object for future use
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump(model, f)
            except Exception:
                model = None

        if model is None:
            # Fallback to exact linear regression parameters from experiment
            model = LinearRegression()
            model.coef_ = np.array([29.61903439])
            model.intercept_ = 161.3267623130613

    return scaler, model

def predict_weight(height_val, scaler, model):
    """
    Predicts weight given height in inches.
    """
    arr = np.array([[float(height_val)]])
    scaled = scaler.transform(arr)
    prediction = model.predict(scaled)
    return float(prediction[0])

# -------------------------------------------------------------
# Streamlit Web GUI
# -------------------------------------------------------------
def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Height to Weight Predictor",
        page_icon="⚖️",
        layout="centered"
    )

    st.title("⚖️ Height & Weight Prediction")
    st.markdown("Predict weight based on height using the **best regression model** (Linear Regression).")

    scaler, model = get_scaler_and_model()

    st.sidebar.header("About Model")
    st.sidebar.info(
        "**Best Model**: Linear Regression\n\n"
        "**Features**: Height (inches)\n\n"
        "**Target**: Weight (pounds / lbs)"
    )

    col1, col2 = st.columns(2)
    with col1:
        unit = st.radio("Height Unit:", ["Inches (in)", "Centimeters (cm)"])
    with col2:
        if unit == "Inches (in)":
            height_input = st.number_input("Enter Height (in):", min_value=30.0, max_value=90.0, value=65.0, step=0.1)
            height_in_inches = height_input
        else:
            height_cm = st.number_input("Enter Height (cm):", min_value=70.0, max_value=230.0, value=165.0, step=0.5)
            height_in_inches = height_cm / 2.54

    if st.button("Predict Weight", type="primary"):
        weight_lbs = predict_weight(height_in_inches, scaler, model)
        weight_kg = weight_lbs * 0.453592

        st.success("### Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Weight (lbs)", f"{weight_lbs:.2f} lbs")
        with res_col2:
            st.metric("Predicted Weight (kg)", f"{weight_kg:.2f} kg")

# -------------------------------------------------------------
# Tkinter Desktop GUI (Fallback or Standalone)
# -------------------------------------------------------------
def run_tkinter():
    import tkinter as tk
    from tkinter import messagebox

    scaler, model = get_scaler_and_model()

    root = tk.Tk()
    root.title("Height to Weight Predictor")
    root.geometry("450x380")
    root.resizable(False, False)
    root.configure(bg="#f4f6f9")

    title_label = tk.Label(
        root, text="Height to Weight Predictor",
        font=("Helvetica", 16, "bold"), bg="#f4f6f9", fg="#2c3e50"
    )
    title_label.pack(pady=15)

    model_label = tk.Label(
        root, text="Model: Linear Regression (Best Model)",
        font=("Helvetica", 10, "italic"), bg="#f4f6f9", fg="#7f8c8d"
    )
    model_label.pack(pady=2)

    frame = tk.Frame(root, bg="#ffffff", bd=1, relief="groove", padx=15, pady=15)
    frame.pack(padx=20, pady=15, fill="both", expand=True)

    unit_var = tk.StringVar(value="in")

    def unit_changed():
        if unit_var.get() == "in":
            unit_lbl.config(text="Height (inches):")
        else:
            unit_lbl.config(text="Height (cm):")

    radio_frame = tk.Frame(frame, bg="#ffffff")
    radio_frame.pack(pady=5)
    tk.Radiobutton(radio_frame, text="Inches (in)", variable=unit_var, value="in", command=unit_changed, bg="#ffffff").pack(side="left", padx=10)
    tk.Radiobutton(radio_frame, text="Centimeters (cm)", variable=unit_var, value="cm", command=unit_changed, bg="#ffffff").pack(side="left", padx=10)

    unit_lbl = tk.Label(frame, text="Height (inches):", font=("Helvetica", 11), bg="#ffffff")
    unit_lbl.pack(pady=5)

    entry_height = tk.Entry(frame, font=("Helvetica", 12), justify="center", width=15)
    entry_height.insert(0, "65.0")
    entry_height.pack(pady=5)

    res_lbs = tk.Label(frame, text="Weight: -- lbs", font=("Helvetica", 12, "bold"), fg="#27ae60", bg="#ffffff")
    res_kg = tk.Label(frame, text="Weight: -- kg", font=("Helvetica", 11), fg="#2980b9", bg="#ffffff")

    def on_predict():
        try:
            val = float(entry_height.get())
            if unit_var.get() == "cm":
                height_inches = val / 2.54
            else:
                height_inches = val
            lbs = predict_weight(height_inches, scaler, model)
            kg = lbs * 0.453592
            res_lbs.config(text=f"Predicted: {lbs:.2f} lbs")
            res_kg.config(text=f"Equivalent: {kg:.2f} kg")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value for height.")

    btn_predict = tk.Button(
        frame, text="Predict Weight", command=on_predict,
        font=("Helvetica", 11, "bold"), bg="#3498db", fg="white",
        activebackground="#2980b9", activeforeground="white", padx=10, pady=5, relief="flat"
    )
    btn_predict.pack(pady=12)

    res_lbs.pack(pady=2)
    res_kg.pack(pady=2)

    root.mainloop()

# -------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    # If running with streamlit runner: `streamlit run app.py`
    # Or if streamlit is explicitly imported by streamlit runner:
    if "streamlit" in sys.modules or (len(sys.argv) > 1 and sys.argv[1] == "--streamlit"):
        run_streamlit()
    else:
        # Check if user invoked directly via python
        # If user passed --tkinter, run tkinter
        if "--tkinter" in sys.argv:
            run_tkinter()
        else:
            try:
                import streamlit
                # If streamlit is installed and user ran `python app.py`, launch streamlit or fallback to tkinter
                from streamlit.web import cli as stcli
                sys.argv = ["streamlit", "run", sys.argv[0]]
                sys.exit(stcli.main())
            except ImportError:
                run_tkinter()
