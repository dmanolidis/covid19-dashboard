import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
import os
from sqlalchemy import create_engine
from scipy.optimize import curve_fit


# Access the heroku database linked to the app
DATABASE_URL = os.environ['DATABASE_URL']
engine = create_engine(DATABASE_URL, connect_args={'sslmode':'require'})


app = dash.Dash(__name__)
app.title = 'COVID-19 Dashboard'
server = app.server


# ===== Functions =====

def transform_export_countries_df(raw):
	
	raw_df = raw.rename(columns={"Country_Region": "country"})
	raw_df.drop(columns=["Lat", "Long"], inplace=True)
	
	# Pivot table for countries as columns and date as index
	final_df = pd.pivot_table(raw_df, columns=["country"])
	final_df = final_df.reset_index()
	final_df = final_df.rename(columns={"index": "date"})
	final_df["date"] = pd.to_datetime(final_df["date"].apply(str))
	final_df = final_df.sort_values("date").reset_index(drop=True)

	return final_df


def total_confirmed_df(confirmed):
	total = {"date": confirmed["date"], "total": confirmed.sum(axis=1)}
	total_df = pd.DataFrame(total)
	return total_df


def new_cases(df, flag_confirmed=True):
	new_cases_df = df.tail(2)
	new_cases_df.diff()
	new_cases_df = new_cases_df.diff()
	new_cases_df = new_cases_df.iloc[1, 1:]
	new_cases_df = pd.DataFrame(new_cases_df).reset_index()

	total_cases = df.tail(1)
	total_cases = total_cases.iloc[0, 1:].reset_index()
	cases = pd.merge(new_cases_df, total_cases, on="country")
	
	if flag_confirmed:
		cases.columns = ["Country", "New Cases", "Total Cases"]
		return cases.sort_values(["New Cases", "Total Cases"], ascending=False)
	else:
		cases.columns = ["Country", "New Deaths", "Total Deaths"]
		return cases.sort_values(["New Deaths", "Total Deaths"], ascending=False)


def get_country_coords(raw):
	country_coords = raw[["Country_Region", "Lat", "Long"]].copy()
	country_coords = country_coords.rename(columns={"Country_Region": "country"})
	return country_coords


def countries_more_than_n(confirmed, deaths, n):
	mask1 = confirmed[confirmed.columns[1:]].apply(max, axis=0) > n
	mask2 = deaths[deaths.columns[1:]].apply(max, axis=0) > 0
	mask = mask1 & mask2
	mask = pd.concat([pd.Series(True, index=["date"]), mask])
	return confirmed.loc[:, mask.values]


def conf_country_df(confirmed, country, n, date_flag):
	df = confirmed[["date", country]].copy()
	df = df[df[country] > n].sort_values("date").copy()
	df = df.rename(columns={country: "country"})

	df["init"] = df[df["country"] > n].iloc[0, 0]
	df["days"] = (df["date"] - df["init"]).dt.days
	df = df.reset_index(drop=True)
	if date_flag:
		 return df[["date", "country"]]
	else:
		 return df[["days", "country"]]


def death_country_df(deaths, country, date_flag):
	df = deaths[["date", country]].copy()
	df = df[df[country] > 0].sort_values("date").copy()
	df = df.rename(columns={country: "country"})

	df["init"] = df[df["country"] > 0].iloc[0, 0]
	df["days"] = (df["date"] - df["init"]).dt.days
	df = df.reset_index(drop=True)
	
	if date_flag:
		return df[["date", "country"]]
	else:
		return df[["days", "country"]]


def new_total_df(total_df):
	total = total_df.copy()
	total["new"] = total["total"].diff()
	return total[["date", "new"]]

	
def new_conf_country_df(confirmed, country):
	df = confirmed[["date", country]].copy()
	df["new"] = df[country].diff()
	
	first_idx = df["new"].loc[df[country] > 0].index[0]
	df = df.iloc[first_idx:, :].reset_index(drop=True)
	
	return df[["date", "new"]]


# ===== Prediction functions =====

def log_func(x, K, a, r):
	return K/(1 + a*np.exp(-r * x))


def exp_func(x, a, r):
	return a*np.exp(-r * x)


def fit_func(xdata, ydata):
	
	# Force curve fit to go through the last point
	sigma = np.ones(len(xdata))
	sigma[-1] = 0.01
	
	# Initial guesses for the curve fit
	K0 = ydata.iloc[-1]
	C0 = ydata.iloc[0]
	a0 = (K0 - C0)/C0
	r0 = 1

	try:
		# First try a logistic curve fit
		popt, pcov = curve_fit(log_func, xdata, ydata, p0=[K0, a0, r0], 
						bounds=([K0, 0, 0], np.inf), sigma=sigma)
		pred = log_func(xdata, *popt)
		flag = "log"
		perr = np.sqrt(np.diag(pcov))
		low_popt = popt - np.array([perr[0], -perr[1], perr[2]])
		high_popt = popt + np.array([perr[0], -perr[1], perr[2]])

	except:
		try:
			# If logistic curve fit fails, try an exponential fit
			popt, pcov = curve_fit(exp_func, xdata, ydata, p0=[a0, -r0],
									bounds=([0, -np.inf], [np.inf, 0]),
									sigma=sigma, maxfev=1000)
			pred = exp_func(xdata, *popt)
			flag = "exp"
			perr = np.sqrt(np.diag(pcov))
			low_popt = popt + np.array([-perr[0], perr[1]])
			high_popt = popt + np.array([perr[0], -perr[1]])

		except:
			# If both fits fail, produce no fit
			popt = []
			pred = []
			pcov = []
			low_popt = []
			high_popt = []
			flag = "nope"
	
	return pred, flag, popt, low_popt, high_popt


def predict_country(country, country_df):
	df = conf_country_df(country_df, country, 100, True).copy()
	xdata = np.array(df.index)
	xdata += 1
	ydata = df["country"]
	# Last point
	y_last = ydata.iloc[-1]

	# Get the curve fit
	pred, flag, popt, low_popt, high_popt = fit_func(xdata, ydata)
	
	# Extension
	last_indx = xdata[-1]
	extension_period = 10
	xdata_extension = np.arange(last_indx+1, last_indx+extension_period)
	xdata_extend = np.append(xdata, xdata_extension)
	
	# Extension for dates
	dates = df["date"]
	init_date = df["date"].iloc[-1]
	extension = pd.date_range(init_date, periods=extension_period)[1:]
	extension = pd.Series(extension)
	extended_dates = pd.concat([dates, extension]).reset_index(drop=True)
	
	if flag=="log":
		pred_full = log_func(xdata_extend, *popt)
		pred_full = np.round(pred_full)
		pred_low = log_func(xdata_extension, *low_popt)
		pred_low = np.round(pred_low)
		pred_low = np.maximum(pred_low, y_last)
		pred_high = log_func(xdata_extension, *high_popt)
		pred_high = np.round(pred_high)

	elif flag=="exp":
		pred_full = exp_func(xdata_extend, *popt)
		pred_full = np.round(pred_full)
		pred_low = exp_func(xdata_extension, *low_popt)
		pred_low = np.round(pred_low)
		pred_low = np.maximum(pred_low, y_last)
		pred_high = exp_func(xdata_extension, *high_popt)
		pred_high = np.round(pred_high)

	else:
		pred_full = []
		pred_low = []
		pred_high = []
	
	return extension, extended_dates, pred_full, pred_low, pred_high, flag


# ===== Global variables =====

colors_palette = {'background_grey': '#f9f9f9', 'pastel_blue': '#19a6db','pastel_red': '#d70b00'}

# Confirmed cases
# Read data from database
conf_raw = pd.read_sql_table('confirmed', engine, index_col=0)
# Transform data
confirmed = transform_export_countries_df(conf_raw)

# Deaths
# Read data from database
death_raw = pd.read_sql_table('deaths', engine, index_col=0)
# Transform data
deaths = transform_export_countries_df(death_raw)

# Population/Death Rate data
# Read data from file
death_rate_pop = pd.read_csv('death_rate_pop.csv', index_col=0)


# Totals
total_confirmed = total_confirmed_df(confirmed)
total_deaths = total_confirmed_df(deaths)

world_total_confirmed = total_confirmed.iloc[-1, -1].astype("int")
world_total_confirmed = '{:,}'.format(world_total_confirmed)
world_total_deaths = total_deaths.iloc[-1, -1].astype("int")
world_total_deaths = '{:,}'.format(world_total_deaths)


# New cases
new_confirmed_cases = new_cases(confirmed)
new_confirmed_deaths = new_cases(deaths, flag_confirmed=False)

new_total_cases = new_total_df(total_confirmed)
new_total_deaths = new_total_df(total_deaths)


# Countries with more than n cases
confirmed_more_100 = countries_more_than_n(confirmed, deaths, 100)
confirmed_more_1 = countries_more_than_n(confirmed, deaths, 1)


# Country coordinates
country_coords = get_country_coords(conf_raw)


# ===== Total Overview tab functions =====

def produce_table(df, table_id):
	return dash_table.DataTable(
						id=table_id,
						columns=[{"name": i, "id": i} for i in df.columns],
						data=df.to_dict('records'),
						style_cell={'fontSize': '1vw', 
									'font-family':'sans-serif',
									'textAlign': 'center',
									'overflow': 'hidden',
									'textOverflow': 'ellipsis',
									'maxWidth': 0},
						
						style_header={'fontSize': '0.8vw',
							'backgroundColor': '#f2f2f2',
							'fontWeight': 'bold',
							'textAlign': 'center',
							'height': 'auto'
						},
						style_table={
							'maxHeight': '600px',
							'overflowY': 'scroll'
						},
						style_as_list_view=True
						)


def graph_totals_tab(df, new_flag, graph_id, feat_dict, layout_dict):
	# This function is used to produce the graphs of the Total Overview tab

	if not new_flag:
		y = df["total"]
	else:
		y = df["new"]
	
	data = {'x': df["date"], 'y': y}
	data.update(feat_dict)
		
	return dcc.Graph(id=graph_id,
				figure={'data': [data],
						'layout': layout_dict},
				style={'height': '34vh'})


def produce_layout_dict(title, log_flag):
	# This function is used to produce the layout of the graphs of the Total Overview tab

	if log_flag:
		yaxis = {'automargin': True, 'type': 'log'}
	else:
		yaxis = {'automargin': True}

	return {'title': {'text': title, 'yanchor': 'top'},
				'paper_bgcolor': colors_palette["background_grey"],
				'plot_bgcolor': colors_palette["background_grey"],
				'margin': dict(l=5, r=5, b=1, t=30),
				'xaxis': {'automargin': True},
				'yaxis': yaxis}


# ===== Styles =====


style_bottom_text = {'textAlign': 'justify', 
						'margin-left': '30%', 
						'margin-right': '30%'}

style_blue = {'font-family': "Arial",
				'font-size': '2.8vw',
				'fontWeight': 'bold',
				'margin-top': '-0.8rem',
				'color': colors_palette["pastel_blue"]}

style_red = {'font-family': "Arial",
				'font-size': '2.8vw',
				'fontWeight': 'bold',
				'margin-top': '-0.8rem',
				'color': colors_palette["pastel_red"]}

style_label = {"margin-bottom": "0rem",
				'margin-top': '0.4rem'}

style_country = {'font-family': "Arial",
						'font-size': '2.2vw',
						'fontWeight': 'bold',
						'margin-top': '-0.8rem',
						'margin-bottom': '0.4rem'}

style_blue_country = {'color': colors_palette["pastel_blue"]}
style_blue_country.update(style_country)
style_red_country = {'color': colors_palette["pastel_red"]}
style_red_country.update(style_country)
style_grey_country = {'color': "grey"}
style_grey_country.update(style_country)

style_container_country_overview = {'textAlign': 'center', 
										'width': '23.625%'}

style_graph_country_overview = {'width': '47.5%', 'height': '35vh'}

style_graph_country_comparison = {'width': '47.5%', 'height': '75vh'}


# ===== Total Overview tab tables and graphs =====

table_confirmed = produce_table(new_confirmed_cases, 'table_confirmed')
table_deaths = produce_table(new_confirmed_deaths, 'table_deaths')

graph1_total = graph_totals_tab(total_confirmed, False, 'graph11', 
									dict(mode='lines', 
										line=dict(color=colors_palette["pastel_blue"],
										width=6)), 
									produce_layout_dict("Total Confirmed Cases", False))

graph1_log_total = graph_totals_tab(total_confirmed, False, 'graph11_log', 
									dict(mode='lines', 
										line=dict(color=colors_palette["pastel_blue"],
										width=6)), 
									produce_layout_dict("Total Confirmed Cases", True))

graph1_new_total = graph_totals_tab(new_total_cases, True, 'graph11_new', 
									dict(type='bar', 
										marker=dict(color=colors_palette["pastel_blue"])), 
									produce_layout_dict("New Confirmed Cases", False))

graph2_total = graph_totals_tab(total_deaths, False, "graph12", 
									dict(mode='lines', 
										line=dict(color=colors_palette["pastel_red"],
										width=6)),
									produce_layout_dict("Total Deaths", False))

graph2_log_total = graph_totals_tab(total_deaths, False, 'graph12_log',  
									dict(mode='lines', 
										line=dict(color=colors_palette["pastel_red"],
										width=6)), 
									produce_layout_dict("Total Deaths", True))

graph2_new_total = graph_totals_tab(new_total_deaths, True, "graph12_new", 
									dict(type='bar', 
										marker=dict(color=colors_palette["pastel_red"])),
									produce_layout_dict("New Deaths", False))


# ===== Map =====

def map_data_df(confirmed_cases_df, confirmed_deaths_df, country_coords):
	new_conf_c = confirmed_cases_df.copy()
	new_conf_c = new_conf_c.rename(columns={"Country": "country"})
	new_conf_d = confirmed_deaths_df.copy()
	new_conf_d = new_conf_d.rename(columns={"Country": "country"})

	df_map = pd.merge(pd.merge(new_conf_c, new_conf_d, on='country'), 
								country_coords,on='country')
	df_map["total_cases_string"] = df_map['Total Cases'].astype("int").apply(\
															lambda x: '{:,}'.format(x))
	df_map["new_cases_string"] = df_map['New Cases'].astype("int").apply(\
															lambda x: '{:,}'.format(x))
	df_map["total_deaths_string"] = df_map['Total Deaths'].astype("int").apply(\
															lambda x: '{:,}'.format(x))
	df_map["new_deaths_string"] = df_map['New Deaths'].astype("int").apply(\
															lambda x: '{:,}'.format(x))

	df_map['text'] = df_map['country'] + '<br>Total Cases: ' + df_map['total_cases_string'] + \
					'<br>New Cases: ' + df_map['new_cases_string'] +\
					'<br>Total Deaths: ' + df_map['total_deaths_string'] + \
					'<br>New Deaths: ' + df_map['new_deaths_string']

	# Manually fixing wrong coordinates
	df_map.loc[df_map["country"]=="Australia", ["Lat", "Long"]] = [-25.2744, 133.7751]
	df_map.loc[df_map["country"]=="Netherlands", ["Lat", "Long"]] = [52.1326, 5.2913]
	df_map.loc[df_map["country"]=="Canada", ["Lat", "Long"]] = [56.1304, -106.3468]
	df_map.loc[df_map["country"]=="China", ["Lat", "Long"]] = [35.8617, 104.1954]
	df_map.loc[df_map["country"]=="Denmark", ["Lat", "Long"]] = [56.2639, 9.5018]
	df_map.loc[df_map["country"]=="France", ["Lat", "Long"]] = [46.2276, 2.2137]
	df_map.loc[df_map["country"]=="United Kingdom", ["Lat", "Long"]] = [55.3781, -3.4360]

	return df_map


map_df = map_data_df(new_confirmed_cases, new_confirmed_deaths, country_coords)

graph_map = dcc.Graph(figure={'data': [dict(
								type="scattergeo",
								lon = map_df['Long'],
								lat = map_df['Lat'],
								text = map_df["text"],
								mode = 'markers',
								marker = dict(
									size = list(map_df['Total Cases']/50 + 30),
									color=colors_palette["pastel_blue"],
									line_width=0.5,
									sizemode = 'area')
									)],
							'layout': dict(geo=dict(
									visible=True, resolution=110,
									showcountries=True, 
									countrycolor="black", 
									projection_type="mercator"),
									margin={"r":0,"t":0,"l":0,"b":0},
									paper_bgcolor=colors_palette["background_grey"],
									plot_bgcolor=colors_palette["background_grey"],
									hoverlabel=dict(font={
										'size': 16,
										'color': "white"}
									   	))},
						style={"height": '75vh'})


# ===== Country Overview =====

def produce_country_overview_container(text, id_label, style1):
	
	return html.Div([html.H6(text, style=style_label),
					html.H1(id=id_label, style=style1)],
						style=style_container_country_overview,														
						className="pretty_container")


def produce_country_overview_graph(id_label):
	return dcc.Graph(id=id_label,
						className="pretty_container",
						style=style_graph_country_overview)


# ===== Layout =====

app.layout = html.Div([
				html.Div([html.H2("COVID-19 Dashboard", 
						style={"margin-bottom": "0px"}),
					
					html.H6("This dashboard monitors the development of the COVID-19 outbreak", 
					 	style={"margin-top": "0px", "margin-bottom": "0px"}),

					html.H6("The data are provided by John Hopkins CSSE", 
					 	style={"margin-top": "0px", "margin-bottom": "40px"})],
					
					style={'textAlign': 'center'},
					id="header"
					),
				
				dcc.Tabs([
					dcc.Tab(id="total_overview", label='Total Overview', 
						children=[
							html.Div([
								html.Div([
									html.H6("Total Confirmed Cases",
										style={"margin-bottom": "0rem"}),
									html.H1(world_total_confirmed, 
										style=style_blue)
									],
									style={'textAlign': 'center'},														
									className="pretty_container",
									),
								
								html.Div(
									[html.H5("Confirmed Cases",
											style={'textAlign': 'center', 
													'margin-top': '-1rem',
													'margin-bottom': '0.2rem'}),
									table_confirmed
									], className="pretty_container",
									)		
								], className="three columns"),
							
								html.Div([
									html.Div([
										html.H6("Total Deaths",
											style={"margin-bottom": "0rem"}),
										html.H1(world_total_deaths, 
											style=style_red)
										],
										style={'textAlign': 'center'},
										className="pretty_container"),
									
									html.Div([
										html.H5("Deaths",
											style={'textAlign': 'center', 
													'margin-top': '-1rem',
													'margin-bottom': '0.2rem'}),
										table_deaths
										], className="pretty_container")	
									], className="three columns"),
										
								html.Div([
									html.Div([
										dcc.Tabs([
											dcc.Tab(label='Linear', 
												children=[
													graph1_total],
													className="custom-tab"),
											dcc.Tab(label='Log', 
												children=[
													graph1_log_total],
													className="custom-tab"),
											dcc.Tab(label='New Cases', 
												children=[
													graph1_new_total],
													className="custom-tab")
											], style={'fontSize': 14,
														'font-family': "Arial"}
												)
										], className="pretty_container"),
											
									html.Div([
										dcc.Tabs([
											dcc.Tab(label='Linear', 
												children=[
													graph2_total],
													className="custom-tab"),
											dcc.Tab(label='Log', 
												children=[
													graph2_log_total],
													className="custom-tab"),
											dcc.Tab(label='New Deaths', 
												children=[
													graph2_new_total],
													className="custom-tab")
											], style={'fontSize': 14,
														'font-family': "Arial"}
												)
										], className="pretty_container"),
											
									], className="six columns"),

						], className="pretty_container"),
					

					dcc.Tab(id="map", label='Map', 
						children=[		
				
							html.Div([graph_map],
									id='graph_map',
									className="pretty_container",
									style={'height': '75vh'}),
							
							html.Div([
								html.P("""This tab shows a map of the countries affected by \
									COVID-19. The size of the bubble is proportional to the total \
									cases of the country. """
									)],	style=style_bottom_text)

							], className="pretty_container"),

					
					dcc.Tab(id="country_overview", label='Country Overview', 
						children=[
							html.Div([
								html.Div([
									dcc.Dropdown(
										id='country_sel',
										options=[{'label': i, 'value': i} for i \
														in confirmed_more_1.columns[1:]],
										value='US'
										)
									],
									style={'width': '25%', 'display': 'inline-block',
											'margin-left': "0.5%"}
									),
				
								html.Div([

									produce_country_overview_container("Total Confirmed Cases", 
										'total_confirmed_country', style_blue_country),

									produce_country_overview_container("New Confirmed Cases", 
										'new_confirmed_country', style_blue_country),

									produce_country_overview_container("Total Deaths", 
										'deaths_country', style_red_country),

									produce_country_overview_container("New Deaths", 
										'new_deaths_country', style_red_country)
									
									], className="row flex-display"),


								html.Div([

									produce_country_overview_container(
										"Total Confirmed Cases per Million", 
										'total_confirmed_country_pop', style_grey_country),

									produce_country_overview_container(
										"Total Deaths per Million", 
										'deaths_country_pop', style_grey_country),

									produce_country_overview_container(
										"Average deaths per day before the outbreak", 
										'country_drpop', style_grey_country),

									produce_country_overview_container(
										"Percent increase of deaths due to COVID-19", 
										'death_rate_country_pop', style_grey_country)

									], className="row flex-display"),

								html.Div([
									produce_country_overview_graph('graph1_country'),
									produce_country_overview_graph('graph2_country')

									], className="row flex-display"),

								html.Div([
									produce_country_overview_graph('graph3_country'),
									produce_country_overview_graph('graph4_country')

									], className="row flex-display")

								]),

							html.Div([
								dcc.Markdown("This tab shows an overview of the confirmed \
									cases and deaths of the selected country. Furthermore, the \
									average number of deaths per day before the COVID-19 outbreak \
									is shown. This is an estimation based on the latest data from \
									the [World Bank Group](https://www.worldbank.org/). Only \
									countries with at least one confirmed \
									case and one death are shown.")											
									], style=style_bottom_text)

							], className="pretty_container"),
					
					dcc.Tab(id="country_prediction", label='Country Prediction', 
						children=[		
							html.Div([
								html.Div([
									dcc.Dropdown(
										id='country_sel_prediction',
										options=[{'label': i, 'value': i} for i \
														in confirmed_more_100.columns[1:]],
										value='US'
										)
									],
									style={'width': '25%', 'display': 'inline-block',
											'margin-left': "0.5%"}
									)
								]),
				
							dcc.Graph(id='graph1_country_prediction',
									className="pretty_container",
									style={'height': '75vh'}),
							
							html.Div([
								html.P("""This tab shows a prediction for the confirmed \
									cases of the selected country. The model used is very simple, \
									not taking into account many factors that can affect the \
									development of the disease, thus, its predicting capability \
									is limited. The prediction starts on the first day the selected \
									country had more than 100 total cases. A logistic curve is fit \
									to the data. If that is not possible (usually that means \
									that the country is on the early stage of exponential \
									growth), an exponential curve is fit to the data. \
									Confidence intervals are also shown."""
									)],	style=style_bottom_text)

							], className="pretty_container"),
					

					dcc.Tab(id="country_comparison", label='Country Comparison', 
						children=[
							html.Div([
								html.Div([
									dcc.Dropdown(
										id='country1',
										options=[{'label': i, 'value': i} for i \
														in confirmed_more_100.columns[1:]],
										value='Italy'
										)
									], style={'width': '25%', 'display': 'inline-block',
											'margin-left': "0.5%"}),
				
								html.Div([
									dcc.Dropdown(
										id='country2',
										options=[{'label': i, 'value': i} for i \
														in confirmed_more_100.columns[1:]],
										value='Spain'
									)
									], style={'width': '25%', 'display': 'inline-block'})
								]),
							
							html.Div([
								dcc.Graph(id='graph1_comparison',
									className="pretty_container",
									style=style_graph_country_comparison),
							
								dcc.Graph(id='graph2_comparison',
									className="pretty_container",
									style=style_graph_country_comparison)
								
								], className="row flex-display"),
							
							html.Div([
								html.P("This tab can be used to compare the data of the two selected \
									countries. The plots are aligned on the first day the countries \
									had more than 100 cases and the first day they had their first \
									death, respectively. Only countries with more than 100 cases and \
									at least one death are shown.")
									], style=style_bottom_text)

						], className="pretty_container")
					
					], style={'fontSize': 20,
						'font-family': "Arial",
						'margin-bottom': '0px'},
						id="tabs"					
					),
				
				html.Div([
					dcc.Markdown("""
					Data by [John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19), \
					[World Bank](https://data.worldbank.org/indicator/sp.dyn.cdrt.in) | \
					[Code](https://github.com/dmanolidis/covid19-dashboard) \
					| Developed by Dimitris Manolidis
					""", style={'textAlign': 'center',
								'margin-top': '0px',
								'margin-left': '30%',
								'margin-right': '30%'},
						id="about")
					])
				])


# ===== Country Overview tab callbacks =====

@app.callback(
	[Output('total_confirmed_country', 'children'),
	 Output('new_confirmed_country', 'children'),
	 Output('deaths_country', 'children'),
	 Output('new_deaths_country', 'children'),
	 Output('total_confirmed_country_pop', 'children'),
	 Output('deaths_country_pop', 'children'),
	 Output('death_rate_country_pop', 'children'),
	 Output('country_drpop', 'children')],
	[Input('country_sel', 'value')])

def update_graphs_countries(country_sel):
	total_conf_country = conf_country_df(confirmed, country_sel, 0, True)
	total_deaths_country = death_country_df(deaths, country_sel, True)
	new_conf_country = new_conf_country_df(confirmed, country_sel)
	new_deaths_country = new_conf_country_df(deaths, country_sel)

	try:
		pop = death_rate_pop.loc[country_sel, "population"]
		drpop = death_rate_pop.loc[country_sel, "death_rate_pop"]
		pop_flag = True

	except:
		pop = np.nan
		drpop = np.nan
		pop_flag = False

	conf_cases = total_conf_country.iloc[-1,1].astype("int")
	conf_cases_str = '{:,}'.format(conf_cases)
	tot_deaths = total_deaths_country.iloc[-1,1].astype("int")
	tot_deaths_str = '{:,}'.format(tot_deaths)
	new_conf_cases = new_conf_country.iloc[-1,1].astype("int")
	new_conf_cases_str = '{:,}'.format(new_conf_cases)
	new_deaths = new_deaths_country.iloc[-1,1].astype("int")
	new_deaths_str = '{:,}'.format(new_deaths)

	if pop_flag:
		conf_cases_pop = conf_cases/pop*1000000
		conf_cases_pop_str = '{:,}'.format(int(round(conf_cases_pop)))
		tot_deaths_pop = tot_deaths/pop*1000000
		tot_deaths_pop_str = '{:.2f}'.format(round(tot_deaths_pop, 2))
		perc_death_rate_pop = new_deaths/drpop*100
		perc_death_rate_pop_str = '{:.2f}'.format(perc_death_rate_pop) + "%"
		drpop_str = '{:,}'.format(int(round(drpop)))

	else:
		conf_cases_pop_str = 'Not available'
		tot_deaths_pop_str = 'Not available'
		perc_death_rate_pop_str = 'Not available'
		drpop_str = 'Not available'



	return conf_cases_str, new_conf_cases_str, tot_deaths_str, new_deaths_str, \
					conf_cases_pop_str, tot_deaths_pop_str, perc_death_rate_pop_str, drpop_str


@app.callback(
	[Output('graph1_country', 'figure'),
	 Output('graph2_country', 'figure'),
	 Output('graph3_country', 'figure'),
	 Output('graph4_country', 'figure')],
	[Input('country_sel', 'value')])

def update_graphs_countries(country_sel):

	if country_sel:
		graph1_df = conf_country_df(confirmed, country_sel, 0, True)
		graph2_df = death_country_df(deaths, country_sel, True)
		graph3_df = new_conf_country_df(confirmed, country_sel)
		graph4_df = new_conf_country_df(deaths, country_sel)

		common_layout = {'paper_bgcolor': colors_palette["background_grey"],
							'plot_bgcolor': colors_palette["background_grey"],
							'margin': dict(l=5, r=5, b=1, t=30),
							'xaxis': {'automargin': True}}

		graph1_layout = {'title': {'text': "Total Confirmed Cases", 'yanchor': 'top'},
									'yaxis': {'automargin': True, 'type': 'log'}}

		graph2_layout = {'title': {'text': "Total Deaths", 'yanchor': 'top'},
									'yaxis': {'automargin': True, 'type': 'log'}}

		graph3_layout = {'title': {'text': "New Confirmed Cases", 'yanchor': 'top'},
									'yaxis': {'automargin': True}}

		graph4_layout = {'title': {'text': "New Deaths", 'yanchor': 'top'},
									'yaxis': {'automargin': True}}

		graph1_layout.update(common_layout)
		graph2_layout.update(common_layout)
		graph3_layout.update(common_layout)
		graph4_layout.update(common_layout)

		graph1 = {'data': [dict(x=graph1_df["date"], y=graph1_df["country"],
					mode='lines+markers', line=dict(color=colors_palette["pastel_blue"], width= 2), 
					marker=dict(color=colors_palette["pastel_blue"], size= 8))],
					'layout': graph1_layout}

		graph2 = {'data': [dict(x=graph2_df["date"], y=graph2_df["country"],
					mode='lines+markers', line=dict(color=colors_palette["pastel_red"], width= 2), 
					marker=dict(color=colors_palette["pastel_red"], size= 8))],
					'layout': graph2_layout}

		graph3 = {'data': [dict(x=graph3_df["date"], y=graph3_df["new"],
					type='bar', 
					marker=dict(color=colors_palette["pastel_blue"]))],
					'layout': graph3_layout}

		graph4 = {'data': [dict(x=graph4_df["date"], y=graph4_df["new"],
					type='bar', 
					marker=dict(color=colors_palette["pastel_red"]))],
					'layout': graph4_layout}
		
		return graph1, graph2, graph3, graph4
	
	else:
		empty_graph = {'data': [dict(x=[],y=[], mode='lines+markers')], 
						'layout': dict(title= "")}
		
		return empty_graph, empty_graph, empty_graph, empty_graph


# ===== Country Prediction tab callbacks =====

@app.callback(
	Output('graph1_country_prediction', 'figure'),
	[Input('country_sel_prediction', 'value')])

def update_graph1_country_prediction(country_1):

	if country_1:
		df_selected = conf_country_df(confirmed_more_100, country_1, 100, True)
		extension, extended_dates, pred_full, pred_low, pred_high, flag = \
						predict_country(country_1, confirmed_more_100)

		data1 = dict(x=df_selected["date"], y=df_selected["country"], 
						name=country_1, mode='lines+markers',
						line={'width': 2, 'color': '#ff7f0e'},
						marker={'size': 10, 'color': '#ff7f0e'})
		
		if flag != "nope":
			data2 = dict(x=extended_dates, y=pred_full, 
							name= "Prediction", mode='lines',
							line={'width': 5,'color': '#1f77b4'})

			data3 = dict(x=list(extension) + list(extension[::-1]), 
							y=list(pred_high) + list(pred_low[::-1]),
							fill="toself", 
							fillcolor="#b3e0ff",
							name="Prediction Interval", 
							mode='lines',
							line={'color': "transparent"},
							legendgroup="group")
			
			data4 = dict(x=list(extension), 
							y=list(pred_low),
							name="", 
							mode='lines',
							line={'width': 0.1,'color': '#b3e0ff'},
							legendgroup="group",
							showlegend=False)

			data5 = dict(x=list(extension), 
							y=list(pred_high),
							name="", 
							mode='lines',
							line={'width': 0.1,'color': '#b3e0ff'},
							legendgroup="group",
							showlegend=False)

			data = [data5, data4, data3, data2, data1]
			if flag == "log":
				title_name = country_1 + ": Logistic Prediction"
			elif flag == "exp":
				title_name = country_1 + ": Exponential Prediction"

		else:
			data = [data1]
			title_name = country_1 + ": No prediction generated"
	
	else:
		data = [dict(x=[], y=[], mode='lines+markers')]
		title_name = ""
		
	return {
		'data': data,
		'layout': dict(title=title_name,
						paper_bgcolor=colors_palette["background_grey"],
						plot_bgcolor=colors_palette["background_grey"],)}


# ===== Country Comparison tab callbacks =====

@app.callback(
	[Output('graph1_comparison', 'figure'),
	Output('graph2_comparison', 'figure')],
	[Input('country1', 'value'),
	Input('country2', 'value')])

def update_graph1_comparison(country_1, country_2):
	common_data = dict(mode='lines+markers',
						line={'width': 2},
						marker={'size': 8})
	
	def produce_data(df, name_label):
		data = dict(x=df["days"], y=df["country"], name=name_label)
		data.update(common_data)
		return data

	if country_1 and country_2:
		c1_conf = conf_country_df(confirmed_more_100, country_1, 100, False)
		c2_conf = conf_country_df(confirmed_more_100, country_2, 100, False)
		data1 = produce_data(c1_conf, country_1)
		data2 = produce_data(c2_conf, country_2)
		data_conf = [data1, data2]
		title_name_conf = "Confirmed Cases:   " + country_1 + " vs " + country_2

		c1_death = death_country_df(deaths, country_1, False)
		c2_death = death_country_df(deaths, country_2, False)
		data1 = produce_data(c1_death, country_1)
		data2 = produce_data(c2_death, country_2)
		data_death = [data1, data2]
		title_name_death = "Deaths:   " + country_1 + " vs " + country_2
	
	elif country_1:
		c1_conf = conf_country_df(confirmed_more_100, country_1, 100, False)
		data1 = produce_data(c1_conf, country_1)
		data_conf = [data1]
		title_name_conf = "Confirmed Cases:   " + country_1

		c1_death = death_country_df(deaths, country_1, False)
		data1 = produce_data(c1_death, country_1)
		data_death = [data1]
		title_name_death = "Deaths:   " + country_1
	
	elif country_2:
		c2_conf = conf_country_df(confirmed_more_100, country_2, 100, False)
		data2 = produce_data(c2_conf, country_2)
		data_conf = [data2]
		title_name_conf = "Confirmed Cases:   " + country_2

		c2_death = death_country_df(deaths, country_2, False)
		data2 = produce_data(c2_death, country_2)
		data_death = [data2]
		title_name_death = "Deaths:   " + country_2
	
	else:
		data_conf = [dict(x=[], y=[], mode='lines+markers')]
		title_name_conf = ""

		data_death = [dict(x=[], y=[], mode='lines+markers')]
		title_name_death = ""
		
	return {'data': data_conf,
			'layout': dict(yaxis={'type': 'log'},
						paper_bgcolor=colors_palette["background_grey"],
						plot_bgcolor=colors_palette["background_grey"],
						title=title_name_conf)},\
			{'data': data_death,
			'layout': dict(yaxis={'type': 'log'},
						paper_bgcolor=colors_palette["background_grey"],
						plot_bgcolor=colors_palette["background_grey"],
						title=title_name_death)}



if __name__ == "__main__":
	app.run_server(debug=False)