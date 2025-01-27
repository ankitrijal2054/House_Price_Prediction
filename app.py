import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Initialize selected features and model variables in session state
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "model" not in st.session_state:
    st.session_state.model = None

# Title and Description
st.title("House Price Predictor")
st.write("Upload your dataset and predict house prices using selected features.")

# File Upload
uploaded_file = st.file_uploader("Upload a clean CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    # Preprocess string columns to convert them into categorical numeric values
    def preprocess_categorical_columns(df):
        categorical_mappings = {}  # Dictionary to store mappings of column -> categories
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category")
            categorical_mappings[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes
        st.write(f"Converted any non integer column  to categorical values.")
        return df, categorical_mappings

    # Apply preprocessing and get the mappings
    data, categorical_mappings = preprocess_categorical_columns(data)
    
    # Select Target and Features
    target_column = st.selectbox("Select Target Column (Price):", options=data.columns)
    feature_columns = st.multiselect(
        "Select Feature Columns:",
        options=data.columns,
        default=list(data.columns.drop(target_column))
    )

    # Feature Selection
    k_features = st.slider("Select Top K Features", min_value=1, max_value=len(feature_columns), value=5)
    
    if st.button("Run Feature Selection and Train Model"):
        # Prepare data
        X = data[feature_columns]
        y = data[target_column]
        
        # Feature Selection
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y)
        st.session_state.selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        st.write(f"Selected Features: {st.session_state.selected_features}")
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Train Model
        st.session_state.model = RandomForestRegressor(random_state=42)
        st.session_state.model.fit(X_train, y_train)
        y_pred = st.session_state.model.predict(X_test)
        
        # Performance Metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Model Performance:\n- RMSE: {rmse:.2f}\n- R²: {r2:.2f}")
        
        # Visualize Predictions
        st.write("Predicted vs Actual:")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(results)

# Use categorical_mappings in the prediction input section
if st.session_state.selected_features and st.session_state.model:
    st.write("Input values for prediction:")
    
    # Initialize session state for input values
    if "input_values" not in st.session_state:
        st.session_state.input_values = {feature: 0.0 for feature in st.session_state.selected_features}
    
    # Create input fields for each selected feature
    for feature in st.session_state.selected_features:
        if feature in categorical_mappings:  # If the feature is categorical
            categories = categorical_mappings[feature]  # Get the mapping for this feature
            st.session_state.input_values[feature] = st.selectbox(
                f"{feature}",
                options=list(categories.keys()),  # Numeric values (keys of the mapping)
                format_func=lambda x: categories[x],  # Display the original category names
                key=f"input_{feature}"
            )
        else:  # Numeric feature
            st.session_state.input_values[feature] = st.number_input(
                f"{feature}",
                value=st.session_state.input_values[feature],
                key=f"input_{feature}"
            )

    
    # Button to make predictions
    if st.button("Predict Price"):
        # Convert session state values into a DataFrame
        input_df = pd.DataFrame([st.session_state.input_values])
        prediction = st.session_state.model.predict(input_df)[0]
        st.success(f"Predicted Price: ${prediction:.2f}")
else:
    st.warning("Please run feature selection and model training before making predictions.")
