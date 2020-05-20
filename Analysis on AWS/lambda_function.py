import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import boto3


# Extracting Job Title, Company Details, Location, Salary and Job Description for Analysis
def extract_job_title(soup, jobs, rows):
    for div in rows:
        for a in div.find_all(name='a', attrs={'data-tn-element': 'jobTitle'}):
            jobs.append(a['title'])
    return (jobs)


def extract_company(soup, companies, rows):
    for div in rows:
        company = div.find_all(name='span', attrs={'class': 'company'})
        if len(company) > 0:
            for b in company:
                companies.append(b.text.strip())
        else:
            sec_try = div.find_all(name='span', attrs={'class': 'result-link-source'})
            for span in sec_try:
                companies.append(span.text.strip())
    return (companies)


def extract_location(soup, locations, rows):
    for div in rows:
        try:
            location_div = div.find(name='div', attrs={'class': 'recJobLoc'})
            locations.append(location_div['data-rc-loc'])
        except:
            locations.append(np.nan)

    return (locations)


def extract_salary(soup, salaries, rows):
    for div in rows:
        try:
            salaries.append(div.find('nobr').text)
        except:
            try:
                div_two = div.find(name='div', attrs={'class': 'salarySnippet'})
                div_three = div_two.find('span')
                salaries.append(div_three.text.strip())
            except:
                salaries.append(np.nan)
    return (salaries)


def extract_description(soup, description, rows):
    spans = soup.findAll('div', attrs={'class': 'summary'})
    for span in spans:
        description.append(span.text.strip())
        # print(span.text.strip())
    return (description)


# Extracting Job Title, Company Details, Location, Salary and Job Description for Analysis
def indeed_scrape():
    jobs = []
    companies = []
    locations = []
    salaries = []
    description = []
    # Number of pages to be scraped = (100*10)+1 
    # where 100 is the actual number of pages scraped. 10 is the offset for each indeed page.
    max_results = 1001

    for start_val in range(0, max_results, 10):
        # url of indeed web page with job title filter set to data scientist.
        page = requests.get('https://www.indeed.com/jobs?q=Data+Scientist&start={}'.format(start_val))
        # ensuring at least 1 second between page extracts.
        time.sleep(1)
        soup = BeautifulSoup(page.text, 'html.parser')
        # Extract div class which contains the information about a single job.
        rows = soup.find_all(name='div', attrs={'class': 'row'})
        job_title = extract_job_title(soup, jobs, rows)
        company_name = extract_company(soup, companies, rows)
        location = extract_location(soup, locations, rows)
        salaries = extract_salary(soup, salaries, rows)
        description = extract_description(soup, description, rows)
        # Create a dataframe from scraped data.
        indeed_df = pd.DataFrame(
            {'company_name': company_name, 'job_title': job_title, 'location': location, 'salaries': salaries,
             'description': description})

    return indeed_df


def preprocess(indeed_data):
    # Indeed webpages may contain multiple job postings of same job. To ensure single data entry per job, duplicate entries are
    # dropped if a job with same location,job title , company name and description is already present.
    indeed_data = indeed_data.drop_duplicates(subset=['location', 'job_title', 'company_name', 'description'],
                                              keep='last').reset_index()

    # Extract the state from location column.
    indeed_data['state'] = np.nan
    for i in range(len(indeed_data['state'])):
        try:
            indeed_data.loc[i, 'state'] = indeed_data.loc[i, 'location'].split(',')[1]
        except:
            pass
    # Group data by state and count the number of jobs available per state.
    no_of_jobs = indeed_data.groupby(['state'])['company_name'].count().reset_index().sort_values(['company_name'])

    # Extract the available max and min salary boundaries for every job posting
    indeed_data['min_salary'] = indeed_data['salaries'].str.split('-').str[0].str.split().str[0].str[1:]
    indeed_data['max_salary'] = indeed_data['salaries'].str.split('-').str[1].str.split().str[0].str[1:]

    for i in range(len(indeed_data['min_salary'])):
        if indeed_data.loc[i, 'min_salary'] is not np.NaN:
            indeed_data.loc[i, 'min_salary'] = str(indeed_data.loc[i, 'min_salary']).replace(',', '')
            indeed_data.loc[i, 'max_salary'] = str(indeed_data.loc[i, 'max_salary']).replace(',', '')
    # Check the salary unit (example: hourly salary/yearly) and convert the available salary to Yearly amount.
    indeed_data['min_salary'] = indeed_data['min_salary'].str.replace('(Indeed est.)', '')
    indeed_data["Suffix"] = indeed_data["salaries"].str.split().str[-1]

    indeed_data['min_salary'] = indeed_data['min_salary'].astype('float')
    indeed_data['max_salary'] = indeed_data['max_salary'].astype('float')

    indeed_data['mean_salary'] = np.nan
    for i in range(len(indeed_data['min_salary'])):
        if (indeed_data.loc[i, 'Suffix'] == 'hour'):
            # Consider full time employee with 40hours/ week , 1 year = 52.1429 weeks.
            indeed_data.loc[i, 'min_salary'] = indeed_data.loc[i, 'min_salary'] * 40 * 52.1429
            indeed_data.loc[i, 'max_salary'] = indeed_data.loc[i, 'max_salary'] * 40 * 52.1429

        # Calculate mean salary from minimum and maximum salary
        if pd.isnull(indeed_data['min_salary'][i]):
            indeed_data.loc[i, 'mean_salary'] = indeed_data['max_salary'][i]
        elif pd.isnull(indeed_data['max_salary'][i]):
            indeed_data.loc[i, 'mean_salary'] = indeed_data['min_salary'][i]
        else:
            indeed_data.loc[i, 'mean_salary'] = (indeed_data['min_salary'][i] + indeed_data['max_salary'][i]) / 2

    # Determine the specialization such as NLP , ML, AI from job title.
    indeed_data = extract_specialization(indeed_data)

    return indeed_data


# Formatting of all graphs
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize': (12, 8)})


def extract_specialization(job_data):
    # Categorizing job titles into specialization type.
    job_data['Job_Title_Category'] = np.nan
    job_data['job_title'] = job_data['job_title'].str.lower()

    job_data.loc[job_data['job_title'].str.contains(
        'data scientist|data science|data science & insights|data science and insights'), 'Job_Title_Category'] = 'Data Scientist'
    job_data.loc[job_data['job_title'].str.contains(
        'analyst|analytics|analysis'), 'Job_Title_Category'] = 'Data Analyst'
    job_data.loc[job_data['job_title'].str.contains(
        'intern|internship|university|graduate|coop|student|co-op'), 'Job_Title_Category'] = 'Data Science Intern/ University Graduate'
    job_data.loc[job_data['job_title'].str.contains(
        'jr|junior|entry level|early career'), 'Job_Title_Category'] = 'Junior Data Scientist'
    job_data.loc[job_data['job_title'].str.contains(
        'sr|senior|phd|research'), 'Job_Title_Category'] = 'Senior Data Scientist'
    job_data.loc[job_data['job_title'].str.contains(
        'machine learning|machine_learning|deep|ai|artificial intelligence'), 'Job_Title_Category'] = 'Machine Learning/ AI/ Deep Learning'
    job_data.loc[job_data['job_title'].str.contains(
        'health|biomedical|bio|bioengineer|bioinformatics|neuro'), 'Job_Title_Category'] = 'Health/ Biomedical Data Science'
    job_data.loc[job_data['job_title'].str.contains(
        'nlp|language'), 'Job_Title_Category'] = 'Natural Language Processing'
    job_data.loc[job_data['job_title'].str.contains(
        'market|quantitative|digital marketing|search|supply chain|payment|advertising'), 'Job_Title_Category'] = 'Data Science-Marketing'
    job_data['Job_Title_Category'] = job_data.Job_Title_Category.replace(np.nan, 'Others', regex=True)

    return job_data

def plot_mean_salary_per_state(indeed_data):
    indeed_data = indeed_data[['mean_salary', 'state']].dropna()
    fig, ax = plt.subplots()
    sns.boxplot(x="state", y="mean_salary", data=indeed_data, ax=ax)
    plt.xlabel("States")
    plt.ylabel("Mean Salary")
    plt.title("Mean Salary per State")
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    save_plot_to_s3('indeed-analysis', 'mean_salary_per_state.png', img_data)


def plot_designation_cnt(indeed_data):
    df = \
    indeed_data.groupby('state').count().sort_values(['Job_Title_Category'], ascending=False).head(15).reset_index()[
        'state']
    jobs_top_state = indeed_data[indeed_data['state'].isin(df)]
    job_category = extract_specialization(jobs_top_state)
    job_category = pd.crosstab(job_category.state, job_category.Job_Title_Category)
    fig, ax = plt.subplots()
    sns.heatmap(job_category, annot=True, fmt="d", ax=ax)
    plt.title("Job Openings per State");
    plt.xlabel("Job Specialization")
    plt.ylabel("States")
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    save_plot_to_s3('indeed-analysis', 'designation_count.png', img_data)


def plot_no_of_jobs_per_state(jobs_data):
    no_of_jobs = jobs_data.groupby(['state']).count().reset_index().sort_values(['company_name'], ascending=False).head(
        15)
    fig, ax = plt.subplots()
    sns.barplot(x="state", y="company_name", data=no_of_jobs, ax=ax)
    plt.xlabel("States")
    plt.ylabel("Number of job postings")
    plt.title("Number of Data Science Jobs per state")
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    save_plot_to_s3('indeed-analysis', 'no_of_jobs_per_state.png', img_data)


def plot_salary_per_specialization(data_preprocessed):
    fig, ax = plt.subplots()
    sns.scatterplot(x='state', y='mean_salary', hue='Job_Title_Category', data=data_preprocessed, ax=ax)
    plt.xlabel("States")
    plt.ylabel("Mean Salary")
    plt.title("Salary distribution per specialization for each state")
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    save_plot_to_s3('indeed-analysis', 'salary_per_specialization.png', img_data)


def save_file_to_s3(bucket, file_name, data):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, file_name)
    obj.put(Body=csv_buffer.getvalue())


def save_plot_to_s3(bucket, file_name, img_data):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    bucket.put_object(Body=img_data, ContentType='image/png', Key=file_name, ACL='public-read')


def lambda_handler(event, context):
    data = indeed_scrape()
    file_name = "daily_jobs_{}".format(time.time())
    print(file_name)
    save_file_to_s3('indeed-jobs', file_name, data)
    data_preprocessed = preprocess(data)
    plot_mean_salary_per_state(data_preprocessed)
    plot_salary_per_specialization(data_preprocessed)
    plot_no_of_jobs_per_state(data_preprocessed)
    plot_designation_cnt(data_preprocessed)
