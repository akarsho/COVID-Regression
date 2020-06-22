# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load COVID-19 Dataset
covid_ds = pd.read_csv('data.csv', sep=',')

# Drop unneeded columns and replace all 'NaN' values with 0
covid_ds = covid_ds.drop(columns=['iso_code', 'continent', 'total_cases_per_million', 'new_cases_per_million',
                                  'total_deaths_per_million', 'new_deaths_per_million', 'new_tests',
                                  'total_tests_per_thousand',
                                  'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
                                  'tests_units',
                                  'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older',
                                  'aged_70_older',
                                  'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate', 'diabetes_prevalence',
                                  'female_smokers',
                                  'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
                                  'life_expectancy', 'total_tests'])
covid_ds.fillna(0, inplace=True)
test_ds = covid_ds[covid_ds["location"] == 'World']
test_ds = test_ds.reset_index(drop=True)
length_df = len(test_ds.index)

# Create quantitative column to represent dates in the column
days = []
for x in range(length_df):
 days.append(x+1)
days_df = pd.DataFrame(days)
days_df.columns = ['days']
print(days_df)
all_ds = pd.concat([days_df, test_ds],axis=1)
print(all_ds.head())

# Seperate dataset into features and labels


