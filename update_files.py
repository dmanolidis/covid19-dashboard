import pandas as pd
from sqlalchemy import create_engine
import os


def create_df(url):
    df = pd.read_csv(url)
    df = df.groupby("Country/Region", as_index=False).sum()
    country_coords = df.groupby("Country/Region", as_index=False).mean()
    df[["Lat", "Long"]] = country_coords[["Lat", "Long"]].copy()
    df = df.rename(columns={"Country/Region": "Country_Region"})
    return df


def update_all_files(url):
    url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'\
    					'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'\
    					'csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'\
    					'csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    confirmed_df = create_df(url_confirmed)
    deaths_df = create_df(url_deaths)
    recovered_df = create_df(url_recovered)
    
    engine = create_engine(url, connect_args={'sslmode':'require'})
    confirmed_df.to_sql('confirmed', engine, if_exists="replace", index=False)
    deaths_df.to_sql('deaths', engine, if_exists="replace", index=False)
    recovered_df.to_sql('recovered', engine, if_exists="replace", index=False)


DATABASE_URL = os.environ['DATABASE_URL']

update_all_files(DATABASE_URL)