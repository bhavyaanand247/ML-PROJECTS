import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# trained ML model
model = joblib.load("model/employee_attrition_model.pkl")

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide"
)

# employee dataset
employee_data = pd.read_csv("data/employee_attrition.csv")
st.markdown("## Employee Details")

age = st.number_input(
    "Age",
    min_value=int(employee_data["Age"].min()),
    max_value=int(employee_data["Age"].max()),
    value=30
)

business_travel = st.selectbox(
    "Business Travel",
    sorted(employee_data["BusinessTravel"].unique())
)
department = st.selectbox(
    "Department",
    sorted(employee_data["Department"].unique())
)

distance_from_home = st.number_input(
    "Distance From Home",
    min_value=int(employee_data["DistanceFromHome"].min()),
    max_value=int(employee_data["DistanceFromHome"].max()),
    value=5
)

st.markdown("## Job & Compensation")
education = st.selectbox(
    "Education Level",
    sorted(employee_data["Education"].unique())
)

# 1-below college, 2- college, 3-bachelor, 4-master, 5-doctor

education_field = st.selectbox(
    "Education Field",
    sorted(employee_data["EducationField"].unique())
)

job_level = st.selectbox(
    "Job Level",
    sorted(employee_data["JobLevel"].unique())
)

job_role = st.selectbox(
    "Job Role",
    sorted(employee_data["JobRole"].unique())
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=int(employee_data["MonthlyIncome"].min()),
    max_value=int(employee_data["MonthlyIncome"].max()),
    value=int(employee_data["MonthlyIncome"].median())
)

overtime = st.selectbox(
    "Overtime",
    sorted(employee_data["OverTime"].unique())
)

st.markdown("## Satisfaction & Work Experience")
environment_satisfaction = st.selectbox(
    "Environment Satisfaction",
    sorted(employee_data["EnvironmentSatisfaction"].unique())
)

job_satisfaction = st.selectbox(
    "Job Satisfaction",
    sorted(employee_data["JobSatisfaction"].unique())
)

job_involvement = st.selectbox(
    "Job Involvement",
    sorted(employee_data["JobInvolvement"].unique())
)

years_at_company = st.number_input(
    "Years at Company",
    min_value=int(employee_data["YearsAtCompany"].min()),
    max_value=int(employee_data["YearsAtCompany"].max()),
    value=int(employee_data["YearsAtCompany"].median())
)

years_in_current_role = st.number_input(
    "Years in Current Role",
    min_value=int(employee_data["YearsInCurrentRole"].min()),
    max_value=int(employee_data["YearsInCurrentRole"].max()),
    value=int(employee_data["YearsInCurrentRole"].median())
)

years_since_promotion = st.number_input(
    "Years Since Last Promotion",
    min_value=int(employee_data["YearsSinceLastPromotion"].min()),
    max_value=int(employee_data["YearsSinceLastPromotion"].max()),
    value=int(employee_data["YearsSinceLastPromotion"].median())
)

years_with_manager = st.number_input(
    "Years With Current Manager",
    min_value=int(employee_data["YearsWithCurrManager"].min()),
    max_value=int(employee_data["YearsWithCurrManager"].max()),
    value=int(employee_data["YearsWithCurrManager"].median())
)

st.markdown("## Additional Employee Details")

gender = st.selectbox(
    "Gender",
    sorted(employee_data["Gender"].unique())
)

marital_status = st.selectbox(
    "Marital Status",
    sorted(employee_data["MaritalStatus"].unique())
)

daily_rate = st.number_input(
    "Daily Rate",
    min_value=int(employee_data["DailyRate"].min()),
    max_value=int(employee_data["DailyRate"].max()),
    value=int(employee_data["DailyRate"].median())
)

hourly_rate = st.number_input(
    "Hourly Rate",
    min_value=int(employee_data["HourlyRate"].min()),
    max_value=int(employee_data["HourlyRate"].max()),
    value=int(employee_data["HourlyRate"].median())
)

monthly_rate = st.number_input(
    "Monthly Rate",
    min_value=int(employee_data["MonthlyRate"].min()),
    max_value=int(employee_data["MonthlyRate"].max()),
    value=int(employee_data["MonthlyRate"].median())
)

num_companies_worked = st.number_input(
    "Number of Companies Worked",
    min_value=int(employee_data["NumCompaniesWorked"].min()),
    max_value=int(employee_data["NumCompaniesWorked"].max()),
    value=int(employee_data["NumCompaniesWorked"].median())
)

total_working_years = st.number_input(
    "Total Working Years",
    min_value=int(employee_data["TotalWorkingYears"].min()),
    max_value=int(employee_data["TotalWorkingYears"].max()),
    value=int(employee_data["TotalWorkingYears"].median())
)

training_times = st.number_input(
    "Training Times Last Year",
    min_value=int(employee_data["TrainingTimesLastYear"].min()),
    max_value=int(employee_data["TrainingTimesLastYear"].max()),
    value=int(employee_data["TrainingTimesLastYear"].median())
)

work_life_balance = st.selectbox(
    "Work-Life Balance",
    sorted(employee_data["WorkLifeBalance"].unique())
)

relationship_satisfaction = st.selectbox(
    "Relationship Satisfaction",
    sorted(employee_data["RelationshipSatisfaction"].unique())
)

stock_option_level = st.selectbox(
    "Stock Option Level",
    sorted(employee_data["StockOptionLevel"].unique())
)

percent_salary_hike = st.number_input(
    "Percent Salary Hike",
    min_value=int(employee_data["PercentSalaryHike"].min()),
    max_value=int(employee_data["PercentSalaryHike"].max()),
    value=int(employee_data["PercentSalaryHike"].median())
)

performance_rating = st.selectbox(
    "Performance Rating",
    sorted(employee_data["PerformanceRating"].unique())
)

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Employee Attrition",
    use_container_width=True
)

if predict_button:

    input_data = pd.DataFrame({
        "Age": [age],
        "BusinessTravel": [business_travel],
        "DailyRate": [daily_rate],
        "Department": [department],
        "DistanceFromHome": [distance_from_home],
        "Education": [education],
        "EducationField": [education_field],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "Gender": [gender],
        "HourlyRate": [hourly_rate],
        "JobInvolvement": [job_involvement],
        "JobLevel": [job_level],
        "JobRole": [job_role],
        "JobSatisfaction": [job_satisfaction],
        "MaritalStatus": [marital_status],
        "MonthlyIncome": [monthly_income],
        "MonthlyRate": [monthly_rate],
        "NumCompaniesWorked": [num_companies_worked],
        "OverTime": [overtime],
        "PercentSalaryHike": [percent_salary_hike],
        "PerformanceRating": [performance_rating],
        "RelationshipSatisfaction": [relationship_satisfaction],
        "StockOptionLevel": [stock_option_level],
        "TotalWorkingYears": [total_working_years],
        "TrainingTimesLastYear": [training_times],
        "WorkLifeBalance": [work_life_balance],
        "YearsAtCompany": [years_at_company],
        "YearsInCurrentRole": [years_in_current_role],
        "YearsSinceLastPromotion": [years_since_promotion],
        "YearsWithCurrManager": [years_with_manager]
    })

    print("Input shape:", input_data.shape)
    print("Input columns:", input_data.columns.tolist())

    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("🔴 HIGH ATTRITION RISK")
    else:
        st.success("🟢 LOW ATTRITION RISK")

    st.metric(
        "Attrition Risk",
        f"{probability * 100:.1f}%"
    )

st.markdown("---")
st.markdown("## 📊 Employee Attrition Analytics")

col1, col2, col3 = st.columns(3)

total_employees = len(employee_data)

attrition_count = (
    employee_data["Attrition"] == "Yes"
).sum()

attrition_rate = (
    attrition_count / total_employees
) * 100

with col1:
    st.metric(
        "Total Employees",
        f"{total_employees:,}"
    )

with col2:
    st.metric(
        "Employees Left",
        f"{attrition_count:,}"
    )

with col3:
    st.metric(
        "Overall Attrition Rate",
        f"{attrition_rate:.1f}%"
    )

st.markdown("### Attrition by Department")

department_data = (
    employee_data
    .groupby("Department")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Attrition Rate")
)

st.bar_chart(
    department_data.set_index("Department")
)

st.markdown("### Attrition by Overtime")

overtime_data = (
    employee_data
    .groupby("OverTime")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Attrition Rate")
)

st.bar_chart(
    overtime_data.set_index("OverTime")
)

st.markdown("### Attrition by Job Role")

role_data = (
    employee_data
    .groupby("JobRole")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

st.bar_chart(role_data)

st.markdown("### Attrition by Job Satisfaction")

satisfaction_data = (
    employee_data
    .groupby("JobSatisfaction")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

st.bar_chart(satisfaction_data)

#1- low, med, high, v high

st.markdown("### Attrition by Age Group")

employee_data["AgeGroup"] = pd.cut(
    employee_data["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "Under 25",
        "25–35",
        "36–45",
        "46–55",
        "55+"
    ]
)

age_data = (
    employee_data
    .groupby("AgeGroup", observed=False)["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

st.bar_chart(age_data)

st.markdown("### Monthly Income vs Attrition")

income_data = (
    employee_data
    .groupby("Attrition")["MonthlyIncome"]
    .mean()
)

st.bar_chart(income_data)

