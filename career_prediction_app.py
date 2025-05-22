import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Career Prediction System", layout="wide")

# Title and description
st.title("Career Prediction System")
st.markdown("### Predict your ideal career path based on your skills and preferences")
st.write("This model uses machine learning to suggest the best career options based on your inputs.")

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('roo_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # Create and train the model
    @st.cache_resource
    def train_model(df):
        # Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Preprocessing for numeric features
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X[numeric_cols])
        
        # Preprocessing for categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_categorical = encoder.fit_transform(X[categorical_cols])
        
        # Combine preprocessed features
        X_processed = np.hstack((X_numeric, X_categorical))
        
        # Feature selection - select top k features
        selector = SelectKBest(f_classif, k=min(50, X_processed.shape[1]))
        X_selected = selector.fit_transform(X_processed, y)
        selected_indices = selector.get_support(indices=True)
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)
        
        # Train Random Forest model with optimized hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,       # Increase from 100 to 200
            max_depth=None,         # Allow trees to grow fully
            min_samples_split=5,    # Minimum samples required to split
            min_samples_leaf=2,     # Minimum samples required at leaf node
            max_features='sqrt',    # Use square root of features for each split
            bootstrap=True,         # Use bootstrap samples
            class_weight='balanced', # Handle class imbalance
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation for more robust accuracy estimation
        cv_scores = cross_val_score(model, X_selected, y_encoded, cv=5)
        
        # Print accuracy to terminal
        print("\n" + "=" * 50)
        print(f"MODEL ACCURACY: {accuracy:.4f} ({accuracy:.2%})")
        print(f"CROSS-VALIDATION ACCURACY: {cv_scores.mean():.4f} ({cv_scores.mean():.2%})")
        print("=" * 50)
        
        # Print detailed classification report to terminal
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        print("=" * 50 + "\n")
        
        # Determine feature importance for input reduction
        if isinstance(model, RandomForestClassifier):
            # For Random Forest, we can get feature importance directly
            feature_importance = model.feature_importances_
        else:
            # For other models, use a simple correlation-based approach
            feature_importance = np.ones(X_selected.shape[1])
        
        # Get the most important features (both numeric and categorical)
        important_features = {}
        
        # For numeric features
        numeric_importance = []
        for i, col in enumerate(numeric_cols):
            # Check if any of the original feature's transformed versions are in the selected features
            importance_sum = 0
            for j in range(len(selected_indices)):
                if j < len(numeric_cols):
                    if i == j:
                        importance_sum += feature_importance[j]
            numeric_importance.append((col, importance_sum))
        
        # Sort by importance and get top features
        numeric_importance.sort(key=lambda x: x[1], reverse=True)
        important_features['numeric'] = [x[0] for x in numeric_importance[:9]]  # Top 9 numeric features
        
        # For categorical features - similar approach but more complex due to one-hot encoding
        cat_importance = []
        for i, col in enumerate(categorical_cols):
            cat_importance.append((col, 1.0))  # Simplified - assign equal importance
        
        cat_importance.sort(key=lambda x: x[1], reverse=True)
        important_features['categorical'] = [x[0] for x in cat_importance[:6]]  # Top 6 categorical features
        
        return model, scaler, encoder, label_encoder, selector, important_features, numeric_cols, categorical_cols, selected_indices
    
    # Train the model
    with st.spinner("Training the model... This might take a moment."):
        model, scaler, encoder, label_encoder, selector, important_features, numeric_cols, categorical_cols, selected_indices = train_model(df)
    
    # Get unique values for categorical fields
    unique_values = {}
    for col in categorical_cols:
        unique_values[col] = df[col].unique().tolist()
    
    # Get salary ranges based on job roles
    salary_data = {
        'Database Developer': {'min': 500000, 'max': 1200000, 'experience_factor': 50000},
        'Portal Administrator': {'min': 450000, 'max': 900000, 'experience_factor': 45000},
        'Systems Security Administrator': {'min': 600000, 'max': 1500000, 'experience_factor': 60000},
        'Business Systems Analyst': {'min': 550000, 'max': 1300000, 'experience_factor': 55000},
        'Software Systems Engineer': {'min': 600000, 'max': 1800000, 'experience_factor': 70000},
        'Business Intelligence Analyst': {'min': 700000, 'max': 1600000, 'experience_factor': 65000},
        'CRM Technical Developer': {'min': 550000, 'max': 1200000, 'experience_factor': 50000},
        'Mobile Applications Developer': {'min': 650000, 'max': 1700000, 'experience_factor': 75000},
        'UX Designer': {'min': 500000, 'max': 1400000, 'experience_factor': 60000},
        'Quality Assurance Associate': {'min': 400000, 'max': 900000, 'experience_factor': 40000},
        'Web Developer': {'min': 450000, 'max': 1300000, 'experience_factor': 55000},
        'Information Security Analyst': {'min': 700000, 'max': 1800000, 'experience_factor': 80000},
        'CRM Business Analyst': {'min': 600000, 'max': 1400000, 'experience_factor': 60000}
    }
    
    # Create sidebar for user inputs - REDUCED NUMBER OF INPUTS
    st.sidebar.title("Enter Your Information")
    
    # Academic percentages - reduced to most important ones
    st.sidebar.subheader("Academic Performance")
    important_academic = [col for col in important_features['numeric'] if 'percentage' in col.lower() or 'acedamic' in col.lower()]
    
    academic_inputs = {}
    for col in important_academic[:5]:  # Limit to top 5 academic fields
        field_name = col.replace('Percentage in ', '').replace('Acedamic percentage in ', '')
        academic_inputs[col] = st.sidebar.slider(field_name, 60, 100, 75)
    
    # Work habits - reduced to most important ones
    st.sidebar.subheader("Skills & Work Habits")
    work_habits = [col for col in important_features['numeric'] if 'percentage' not in col.lower() and 'acedamic' not in col.lower()]
    
    work_inputs = {}
    for col in work_habits[:3]:  # Limit to top 3 work habit fields
        work_inputs[col] = st.sidebar.slider(col, 1, 10, 5) if 'rating' in col.lower() or 'points' in col.lower() else st.sidebar.slider(col, 0, 12, 8)
    
    # Personal traits - reduced to most important ones
    st.sidebar.subheader("Personal Traits & Interests")
    important_categorical = important_features['categorical'][:6]  # Top 6 categorical features
    
    categorical_inputs = {}
    for col in important_categorical:
        if col in ['can work long time before system?', 'self-learning capability?', 'Extra-courses did', 
                  'talenttests taken?', 'olympiads?', 'worked in teams ever?', 'Introvert']:
            categorical_inputs[col] = st.sidebar.selectbox(col, ["yes", "no"])
        elif col in ['reading and writing skills', 'memory capability score']:
            categorical_inputs[col] = st.sidebar.selectbox(col, ["poor", "medium", "excellent"])
        elif col in ['Management or Technical']:
            categorical_inputs[col] = st.sidebar.selectbox(col, ["Management", "Technical"])
        elif col in ['hard/smart worker']:
            categorical_inputs[col] = st.sidebar.selectbox(col, ["hard worker", "smart worker"])
        elif col in ['Gentle or Tuff behaviour?']:
            categorical_inputs[col] = st.sidebar.selectbox(col, ["gentle", "stubborn"])
        else:
            categorical_inputs[col] = st.sidebar.selectbox(col, unique_values.get(col, ['none']))
    
    # Experience for salary calculation
    experience_years = st.sidebar.slider("Years of Experience", 0, 15, 2)
    
    # Predict button
    if st.sidebar.button("Predict Career"):
        # Create input data frame with all original columns (will fill with defaults for non-important ones)
        input_data = pd.DataFrame(columns=numeric_cols + categorical_cols)
        
        # Fill with default values first
        for col in numeric_cols:
            input_data[col] = [75 if 'percentage' in col.lower() else 5]  # Default values
        
        for col in categorical_cols:
            if col in ['can work long time before system?', 'self-learning capability?', 'Extra-courses did']:
                input_data[col] = ['yes']
            elif col in ['reading and writing skills', 'memory capability score']:
                input_data[col] = ['medium']
            elif col in unique_values:
                input_data[col] = [unique_values[col][0]]
            else:
                input_data[col] = ['none']
        
        # Update with user inputs
        for col, value in {**academic_inputs, **work_inputs}.items():
            input_data[col] = [value]
            
        for col, value in categorical_inputs.items():
            input_data[col] = [value]
        
        # Preprocess input data
        input_numeric = scaler.transform(input_data[numeric_cols])
        input_categorical = encoder.transform(input_data[categorical_cols])
        input_processed = np.hstack((input_numeric, input_categorical))
        
        # Apply feature selection
        input_selected = selector.transform(input_processed)
        
        # Make prediction
        prediction_encoded = model.predict(input_selected)
        prediction = label_encoder.inverse_transform(prediction_encoded)
        
        # Get top 3 predictions with probabilities
        probabilities = model.predict_proba(input_selected)[0]
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_jobs = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        # Display results
        st.header("Career Prediction Results")
        
        # Main prediction
        st.subheader("Your Ideal Career Match")
        st.markdown(f"### 🏆 {prediction[0]}")
        
        # Calculate salary based on experience
        if prediction[0] in salary_data:
            salary_info = salary_data[prediction[0]]
            base_min = salary_info['min']
            base_max = salary_info['max']
            exp_factor = salary_info['experience_factor']
            
            # Adjust salary based on experience
            min_salary = base_min + (experience_years * exp_factor)
            max_salary = base_max + (experience_years * exp_factor)
            
            st.markdown(f"**Estimated Salary Range:** ₹{min_salary:,} - ₹{max_salary:,} per annum")
            st.markdown(f"**Experience Considered:** {experience_years} years")
        else:
            st.markdown("Salary information not available for this role.")
        
        # Display top 3 matches
        st.subheader("Top 3 Career Matches")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"### 1. {top_3_jobs[0]}")
            st.progress(float(top_3_probs[0]))
            st.markdown(f"Match: {top_3_probs[0]*100:.1f}%")
            if top_3_jobs[0] in salary_data:
                salary_info = salary_data[top_3_jobs[0]]
                min_salary = salary_info['min'] + (experience_years * salary_info['experience_factor'])
                max_salary = salary_info['max'] + (experience_years * salary_info['experience_factor'])
                st.markdown(f"₹{min_salary:,} - ₹{max_salary:,} p.a.")
        
        with col2:
            if len(top_3_jobs) > 1:
                st.markdown(f"### 2. {top_3_jobs[1]}")
                st.progress(float(top_3_probs[1]))
                st.markdown(f"Match: {top_3_probs[1]*100:.1f}%")
                if top_3_jobs[1] in salary_data:
                    salary_info = salary_data[top_3_jobs[1]]
                    min_salary = salary_info['min'] + (experience_years * salary_info['experience_factor'])
                    max_salary = salary_info['max'] + (experience_years * salary_info['experience_factor'])
                    st.markdown(f"₹{min_salary:,} - ₹{max_salary:,} p.a.")
        
        with col3:
            if len(top_3_jobs) > 2:
                st.markdown(f"### 3. {top_3_jobs[2]}")
                st.progress(float(top_3_probs[2]))
                st.markdown(f"Match: {top_3_probs[2]*100:.1f}%")
                if top_3_jobs[2] in salary_data:
                    salary_info = salary_data[top_3_jobs[2]]
                    min_salary = salary_info['min'] + (experience_years * salary_info['experience_factor'])
                    max_salary = salary_info['max'] + (experience_years * salary_info['experience_factor'])
                    st.markdown(f"₹{min_salary:,} - ₹{max_salary:,} p.a.")
        
        # Career insights
        st.subheader("Career Insights")
        
        # Job descriptions
        job_descriptions = {
            'Database Developer': "Designs, implements, and maintains database systems. Works with SQL, database architecture, and ensures data integrity and performance.",
            'Portal Administrator': "Manages web portals, handles user access, content management, and ensures portal functionality and security.",
            'Systems Security Administrator': "Protects computer systems from threats, implements security measures, and monitors networks for security breaches.",
            'Business Systems Analyst': "Analyzes business needs and processes to recommend IT solutions that improve efficiency and effectiveness.",
            'Software Systems Engineer': "Designs, develops, and maintains software systems, focusing on system architecture and integration.",
            'Business Intelligence Analyst': "Analyzes data to help organizations make better business decisions, creates reports and dashboards.",
            'CRM Technical Developer': "Develops and customizes Customer Relationship Management systems to meet business requirements.",
            'Mobile Applications Developer': "Creates applications for mobile devices, focusing on user experience and functionality.",
            'UX Designer': "Designs user interfaces and experiences for websites and applications, focusing on usability and user satisfaction.",
            'Quality Assurance Associate': "Tests software to identify bugs and ensure quality, develops test plans and procedures.",
            'Web Developer': "Creates and maintains websites, working with programming languages, frameworks, and design principles.",
            'Information Security Analyst': "Protects organizations' computer systems and networks from cyber attacks and data breaches.",
            'CRM Business Analyst': "Analyzes customer relationship management needs and processes to improve customer interactions and business outcomes."
        }
        
        if prediction[0] in job_descriptions:
            st.markdown(f"**About {prediction[0]}:** {job_descriptions[prediction[0]]}")
        
        # Skills to develop
        st.markdown("### Skills to Develop")
        
        # Define key skills for each role
        key_skills = {
            'Database Developer': ["SQL", "Database Design", "Data Modeling", "ETL", "Performance Tuning"],
            'Portal Administrator': ["Content Management Systems", "User Management", "Web Technologies", "Security Protocols"],
            'Systems Security Administrator': ["Network Security", "Firewall Configuration", "Security Auditing", "Threat Analysis"],
            'Business Systems Analyst': ["Requirements Analysis", "Process Modeling", "Business Process Improvement", "Stakeholder Management"],
            'Software Systems Engineer': ["System Architecture", "Software Development", "Integration", "Testing", "Deployment"],
            'Business Intelligence Analyst': ["Data Analysis", "SQL", "Data Visualization", "Statistical Analysis", "Reporting"],
            'CRM Technical Developer': ["CRM Platforms", "Customization", "Integration", "Business Process Automation"],
            'Mobile Applications Developer': ["Mobile Frameworks", "UI/UX Design", "API Integration", "Cross-platform Development"],
            'UX Designer': ["User Research", "Wireframing", "Prototyping", "Usability Testing", "Visual Design"],
            'Quality Assurance Associate': ["Test Planning", "Test Automation", "Bug Tracking", "Quality Standards"],
            'Web Developer': ["HTML/CSS", "JavaScript", "Web Frameworks", "Responsive Design", "API Integration"],
            'Information Security Analyst': ["Vulnerability Assessment", "Security Tools", "Incident Response", "Risk Management"],
            'CRM Business Analyst': ["CRM Systems", "Business Analysis", "Process Improvement", "Requirements Gathering"]
        }
        
        if prediction[0] in key_skills:
            skills = key_skills[prediction[0]]
            cols = st.columns(len(skills))
            for i, skill in enumerate(skills):
                with cols[i]:
                    st.markdown(f"**{skill}**")
        
        # Education recommendations
        st.markdown("### Recommended Education/Certifications")
        
        education_recs = {
            'Database Developer': ["Database Administration Certification", "SQL Certification", "Data Engineering Courses"],
            'Portal Administrator': ["Content Management System Certification", "Web Administration Courses"],
            'Systems Security Administrator': ["CISSP", "Security+ Certification", "Network Security Courses"],
            'Business Systems Analyst': ["Business Analysis Certification", "Process Improvement Courses"],
            'Software Systems Engineer': ["Software Engineering Degree", "System Architecture Certification"],
            'Business Intelligence Analyst': ["Data Analysis Certification", "Business Intelligence Tools Training"],
            'CRM Technical Developer': ["CRM Platform Certification", "Business Process Management Courses"],
            'Mobile Applications Developer': ["Mobile Development Certification", "UI/UX Design Courses"],
            'UX Designer': ["UX Design Certification", "User Research Courses", "Visual Design Training"],
            'Quality Assurance Associate': ["Quality Assurance Certification", "Test Automation Courses"],
            'Web Developer': ["Web Development Certification", "Frontend/Backend Framework Courses"],
            'Information Security Analyst': ["Cybersecurity Certification", "Ethical Hacking Courses"],
            'CRM Business Analyst': ["CRM Certification", "Business Analysis Courses"]
        }
        
        if prediction[0] in education_recs:
            recs = education_recs[prediction[0]]
            for rec in recs:
                st.markdown(f"- {rec}")

else:
    st.error("Failed to load the dataset. Please check if the file exists and is in the correct format.")
