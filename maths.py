import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="House Price Prediction with Uncertainty",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Prediction with Uncertainty Quantification")
st.markdown("""
This application demonstrates **predictive modeling with confidence and prediction intervals** 
using multiple linear regression for house price estimation.
""")

# Sidebar
st.sidebar.header("📊 Configuration")
st.sidebar.markdown("---")

# Generate synthetic data
@st.cache_data
def generate_house_data(n_samples=200, random_seed=42):
    """Generate synthetic house price data"""
    np.random.seed(random_seed)
    
    data = {
        'area': np.random.uniform(500, 4000, n_samples),
        'bedrooms': np.random.randint(2, 6, n_samples),
        'age': np.random.uniform(0, 50, n_samples),
        'location': np.random.choice([0, 1], n_samples)  # 0: suburban, 1: urban
    }
    
    # True relationship with noise
    base_price = (50000 + 
                  data['area'] * 150 + 
                  data['bedrooms'] * 25000 - 
                  data['age'] * 1000 + 
                  data['location'] * 50000)
    
    noise = np.random.normal(0, 50000, n_samples)
    data['price'] = np.maximum(base_price + noise, 50000)
    
    df = pd.DataFrame(data)
    df['location_type'] = df['location'].map({0: 'Suburban', 1: 'Urban'})
    
    return df

# Linear Regression class with uncertainty quantification
class LinearRegressionWithUncertainty:
    def __init__(self):
        self.beta = None
        self.X_train = None
        self.y_train = None
        self.residual_std_error = None
        self.XtX_inv = None
        self.n = None
        self.p = None
        
    def fit(self, X, y):
        """Fit linear regression using normal equation"""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        self.X_train = X_with_intercept
        self.y_train = y
        self.n = len(y)
        self.p = X_with_intercept.shape[1]
        
        # Normal equation: β = (X'X)^(-1)X'y
        XtX = X_with_intercept.T @ X_with_intercept
        self.XtX_inv = np.linalg.inv(XtX)
        Xty = X_with_intercept.T @ y
        self.beta = self.XtX_inv @ Xty
        
        # Calculate residual standard error
        y_pred = X_with_intercept @ self.beta
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        self.residual_std_error = np.sqrt(rss / (self.n - self.p))
        
        return self
    
    def predict(self, X, alpha=0.05):
        """Predict with confidence and prediction intervals"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        predictions = X_with_intercept @ self.beta
        
        # t-value for confidence level
        t_value = stats.t.ppf(1 - alpha/2, self.n - self.p)
        
        # Calculate standard errors for each prediction
        confidence_intervals = []
        prediction_intervals = []
        
        for x in X_with_intercept:
            # Standard error for confidence interval (mean response)
            se_fit = self.residual_std_error * np.sqrt(x @ self.XtX_inv @ x)
            ci_lower = predictions[len(confidence_intervals)] - t_value * se_fit
            ci_upper = predictions[len(confidence_intervals)] + t_value * se_fit
            confidence_intervals.append((ci_lower, ci_upper))
            
            # Standard error for prediction interval (individual response)
            se_pred = self.residual_std_error * np.sqrt(1 + x @ self.XtX_inv @ x)
            pi_lower = predictions[len(prediction_intervals)] - t_value * se_pred
            pi_upper = predictions[len(prediction_intervals)] + t_value * se_pred
            prediction_intervals.append((pi_lower, pi_upper))
        
        return predictions, confidence_intervals, prediction_intervals
    
    def get_coefficients(self):
        """Return model coefficients"""
        return self.beta

# Sidebar parameters
n_samples = st.sidebar.slider("Number of Samples", 100, 500, 200, 50)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 5)
confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95, 1)
random_seed = st.sidebar.number_input("Random Seed", 0, 100, 42)

# Generate data
df = generate_house_data(n_samples, random_seed)

# Train/test split
train_size = int(len(df) * (1 - test_size/100))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Prepare features and target
feature_cols = ['area', 'bedrooms', 'age', 'location']
X_train = train_df[feature_cols].values
y_train = train_df['price'].values
X_test = test_df[feature_cols].values
y_test = test_df['price'].values

# Train model
model = LinearRegressionWithUncertainty()
model.fit(X_train, y_train)

# Make predictions
alpha = 1 - confidence_level/100
y_pred_train, ci_train, pi_train = model.predict(X_train, alpha)
y_pred_test, ci_test, pi_test = model.predict(X_test, alpha)

# Calculate metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Calculate prediction interval coverage
pi_coverage = np.mean([
    (y_test[i] >= pi_test[i][0]) and (y_test[i] <= pi_test[i][1])
    for i in range(len(y_test))
])

# Metrics display
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("R² Score (Train)", f"{r2_train:.3f}")
with col2:
    st.metric("R² Score (Test)", f"{r2_test:.3f}")
with col3:
    st.metric("RMSE", f"${rmse_test:,.0f}")
with col4:
    st.metric("MAE", f"${mae_test:,.0f}")
with col5:
    st.metric("PI Coverage", f"{pi_coverage*100:.1f}%")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Model Overview", 
    "📊 Predictions & Intervals", 
    "🎯 Individual Prediction", 
    "📉 Residual Analysis",
    "📚 Statistical Theory"
])

# Tab 1: Model Overview
with tab1:
    st.header("Model Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Equation")
        coef = model.get_coefficients()
        st.latex(r"\text{Price} = \beta_0 + \beta_1 \cdot \text{Area} + \beta_2 \cdot \text{Bedrooms} + \beta_3 \cdot \text{Age} + \beta_4 \cdot \text{Location}")
        
        st.markdown(f"""
        **Coefficients:**
        - Intercept (β₀): ${coef[0]:,.2f}
        - Area (β₁): ${coef[1]:,.2f} per sq ft
        - Bedrooms (β₂): ${coef[2]:,.2f}
        - Age (β₃): ${coef[3]:,.2f} per year
        - Location (β₄): ${coef[4]:,.2f} (urban premium)
        
        **Model Statistics:**
        - Residual Standard Error: ${model.residual_std_error:,.2f}
        - Degrees of Freedom: {model.n - model.p}
        - Training Samples: {len(X_train)}
        - Test Samples: {len(X_test)}
        """)
    
    with col2:
        st.subheader("Feature Distributions")
        fig = px.box(train_df, y=feature_cols, 
                     title="Feature Value Distributions (Training Set)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_train, y=y_pred_train,
            mode='markers',
            name='Training Data',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[y_train.min(), y_train.max()],
            y=[y_train.min(), y_train.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="Training Set",
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred_test,
            mode='markers',
            name='Test Data',
            marker=dict(color='green', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="Test Set",
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Predictions & Intervals
with tab2:
    st.header("Predictions with Uncertainty Intervals")
    
    st.info("""
    **Key Concepts:**
    - 🟢 **Confidence Interval (CI)**: Uncertainty about the *mean* prediction (narrower)
    - 🟣 **Prediction Interval (PI)**: Uncertainty for *individual* predictions (wider)
    - The PI accounts for both model uncertainty AND natural variation in individual observations
    """)
    
    # Sort by area for better visualization
    sort_idx = np.argsort(test_df['area'].values)
    x_plot = test_df['area'].values[sort_idx]
    y_plot = y_test[sort_idx]
    y_pred_plot = y_pred_test[sort_idx]
    ci_plot = [ci_test[i] for i in sort_idx]
    pi_plot = [pi_test[i] for i in sort_idx]
    
    fig = go.Figure()
    
    # Prediction Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_plot, x_plot[::-1]]),
        y=np.concatenate([[pi[1] for pi in pi_plot], 
                         [pi[0] for pi in pi_plot[::-1]]]),
        fill='toself',
        fillcolor='rgba(147, 51, 234, 0.2)',
        line=dict(color='rgba(147, 51, 234, 0)'),
        name=f'{confidence_level}% Prediction Interval',
        showlegend=True
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_plot, x_plot[::-1]]),
        y=np.concatenate([[ci[1] for ci in ci_plot], 
                         [ci[0] for ci in ci_plot[::-1]]]),
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.3)',
        line=dict(color='rgba(34, 197, 94, 0)'),
        name=f'{confidence_level}% Confidence Interval',
        showlegend=True
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_pred_plot,
        mode='lines',
        name='Predicted Price',
        line=dict(color='blue', width=3)
    ))
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode='markers',
        name='Actual Price',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title=f"House Prices with {confidence_level}% Confidence & Prediction Intervals",
        xaxis_title="Area (sq ft)",
        yaxis_title="Price ($)",
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interval widths comparison
    st.subheader("Interval Width Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        ci_widths = [ci[1] - ci[0] for ci in ci_test]
        st.metric("Average CI Width", f"${np.mean(ci_widths):,.0f}")
    
    with col2:
        pi_widths = [pi[1] - pi[0] for pi in pi_test]
        st.metric("Average PI Width", f"${np.mean(pi_widths):,.0f}")
    
    st.markdown(f"""
    The prediction interval is approximately **{np.mean(pi_widths)/np.mean(ci_widths):.2f}x wider** 
    than the confidence interval because it accounts for individual observation variability.
    """)

# Tab 3: Individual Prediction
with tab3:
    st.header("🎯 Make a Prediction with Uncertainty")
    
    st.markdown("Adjust the house features to see the predicted price with uncertainty bounds:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_area = st.slider("Area (sq ft)", 500, 4000, 2000, 100)
        pred_bedrooms = st.slider("Number of Bedrooms", 2, 5, 3)
    
    with col2:
        pred_age = st.slider("Age (years)", 0, 50, 10)
        pred_location = st.selectbox("Location", ["Suburban", "Urban"])
    
    # Make prediction
    pred_location_num = 1 if pred_location == "Urban" else 0
    X_new = np.array([[pred_area, pred_bedrooms, pred_age, pred_location_num]])
    y_new_pred, ci_new, pi_new = model.predict(X_new, alpha)
    
    st.markdown("---")
    
    # Display prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Predicted Price")
        st.markdown(f"<h1 style='color: #4F46E5;'>${y_new_pred[0]:,.0f}</h1>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### 🟢 {confidence_level}% Confidence Interval")
        st.markdown("*For average price*")
        st.markdown(f"**${ci_new[0][0]:,.0f}** to **${ci_new[0][1]:,.0f}**")
        st.markdown(f"*Width: ${ci_new[0][1] - ci_new[0][0]:,.0f}*")
    
    with col3:
        st.markdown(f"### 🟣 {confidence_level}% Prediction Interval")
        st.markdown("*For individual house*")
        st.markdown(f"**${pi_new[0][0]:,.0f}** to **${pi_new[0][1]:,.0f}**")
        st.markdown(f"*Width: ${pi_new[0][1] - pi_new[0][0]:,.0f}*")
    
    # Visualization
    fig = go.Figure()
    
    intervals = [
        ('Prediction', y_new_pred[0], y_new_pred[0], '#4F46E5'),
        ('Confidence Interval', ci_new[0][0], ci_new[0][1], '#22C55E'),
        ('Prediction Interval', pi_new[0][0], pi_new[0][1], '#9333EA')
    ]
    
    for name, lower, upper, color in intervals:
        fig.add_trace(go.Scatter(
            x=[lower, upper],
            y=[name, name],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=10),
            marker=dict(size=15, symbol='line-ns-open')
        ))
    
    fig.update_layout(
        title="Prediction with Uncertainty Bounds",
        xaxis_title="Price ($)",
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.success(f"""
    **Interpretation:**
    - We are {confidence_level}% confident that the *average* price for houses with these characteristics 
      is between **${ci_new[0][0]:,.0f}** and **${ci_new[0][1]:,.0f}**.
    - For any *individual* house with these characteristics, there's a {confidence_level}% chance 
      its price falls between **${pi_new[0][0]:,.0f}** and **${pi_new[0][1]:,.0f}**.
    - The prediction interval is wider because it accounts for natural variation between individual houses.
    """)

# Tab 4: Residual Analysis
with tab4:
    st.header("Residual Analysis")
    
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Residuals vs Fitted Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred_test, y=residuals_test,
            mode='markers',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Random scatter around zero indicates good model fit")
    
    with col2:
        st.subheader("Q-Q Plot")
        from scipy import stats as sp_stats
        (osm, osr), (slope, intercept, r) = sp_stats.probplot(residuals_test, dist="norm")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=osm, y=osr,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=osm, y=slope * osm + intercept,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points along the line indicate normally distributed residuals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Residual Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals_test,
            nbinsx=30,
            name='Residuals',
            marker_color='blue',
            opacity=0.7
        ))
        fig.update_layout(
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Residual Statistics")
        st.markdown(f"""
        - **Mean**: ${np.mean(residuals_test):,.2f}
        - **Std Dev**: ${np.std(residuals_test):,.2f}
        - **Min**: ${np.min(residuals_test):,.2f}
        - **Max**: ${np.max(residuals_test):,.2f}
        - **Skewness**: {sp_stats.skew(residuals_test):.3f}
        - **Kurtosis**: {sp_stats.kurtosis(residuals_test):.3f}
        
        **Normality Test (Shapiro-Wilk):**
        """)
        stat, p_value = sp_stats.shapiro(residuals_test[:50])  # Use subset for test
        st.markdown(f"- p-value: {p_value:.4f}")
        if p_value > 0.05:
            st.success("✓ Residuals appear normally distributed (p > 0.05)")
        else:
            st.warning("⚠ Residuals may not be normally distributed (p < 0.05)")

# Tab 5: Statistical Theory
with tab5:
    st.header("📚 Statistical Theory & Formulas")
    
    st.markdown("""
    ## 1. Linear Regression Model
    
    The multiple linear regression model is expressed as:
    """)
    
    st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon")
    st.latex(r"\text{where } \epsilon \sim N(0, \sigma^2)")
    
    st.markdown("""
    ## 2. Parameter Estimation (Ordinary Least Squares)
    
    The coefficients are estimated using the **Normal Equation**:
    """)
    
    st.latex(r"\hat{\beta} = (X^T X)^{-1} X^T y")
    
    st.markdown("""
    This minimizes the sum of squared residuals (RSS):
    """)
    
    st.latex(r"RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")
    
    st.markdown("""
    ## 3. Residual Standard Error
    
    Estimates the standard deviation of the error term:
    """)
    
    st.latex(r"RSE = \hat{\sigma} = \sqrt{\frac{RSS}{n-p}}")
    
    st.markdown("""
    where n is the number of observations and p is the number of parameters.
    
    ## 4. Confidence Interval for Mean Response
    
    The confidence interval for the **average** prediction at x₀:
    """)
    
    st.latex(r"\hat{y}_0 \pm t_{\alpha/2, n-p} \cdot SE(\hat{y}_0)")
    st.latex(r"SE(\hat{y}_0) = \hat{\sigma} \sqrt{x_0^T (X^T X)^{-1} x_0}")
    
    st.markdown("""
    - Quantifies uncertainty about the **mean** response
    - Gets narrower as sample size increases
    - Narrower than prediction interval
    
    ## 5. Prediction Interval for Individual Response
    
    The prediction interval for an **individual** observation at x₀:
    """)
    
    st.latex(r"\hat{y}_0 \pm t_{\alpha/2, n-p} \cdot SE_{pred}")
    st.latex(r"SE_{pred} = \hat{\sigma} \sqrt{1 + x_0^T (X^T X)^{-1} x_0}")
    
    st.markdown("""
    - Accounts for both estimation uncertainty AND individual variation
    - The "+1" term represents additional variance from individual observations
    - Always wider than confidence interval
    
    ## 6. Key Differences
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Confidence Interval:**
        - For mean/average prediction
        - Answers: "Where is the true mean?"
        - Narrower interval
        - SE includes only model uncertainty
        """)
    
    with col2:
        st.warning("""
        **Prediction Interval:**
        - For individual prediction
        - Answers: "Where will a new observation fall?"
        - Wider interval
        - SE includes model + individual uncertainty
        """)
    
    st.markdown("""
    ## 7. Model Evaluation Metrics
    """)
    
    st.latex(r"R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
    
    st.markdown("""
    - **R²**: Proportion of variance explained (0 to 1, higher is better)
    - **RMSE**: Root Mean Squared Error (same units as target, lower is better)
    - **MAE**: Mean Absolute Error (robust to outliers, lower is better)
    - **PI Coverage**: Percentage of test observations within prediction intervals (should ≈ confidence level)
    
    ## 8. Assumptions of Linear Regression
    
    1. **Linearity**: Relationship between X and Y is linear
    2. **Independence**: Observations are independent
    3. **Homoscedasticity**: Constant variance of errors
    4. **Normality**: Errors are normally distributed
    5. **No multicollinearity**: Predictors are not highly correlated
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>House Price Prediction with Uncertainty Quantification</p>
        <p>Built with Streamlit | Statistical Modeling & Predictive Intervals</p>
    </div>
    """, unsafe_allow_html=True)