import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import os
from sqlalchemy import create_engine

DATABASE_URL = os.environ['DATABASE_URL']
engine = create_engine(DATABASE_URL, connect_args={'sslmode':'require'})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
app.title = 'COVID-19 Dashboard'
server = app.server


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


# Prediction
from scipy.optimize import curve_fit


def log_func(x, K, a, r):
    return K/(1 + a*np.exp(-r * x))


def exp_func(x, a, r):
    return a*np.exp(-r * x)


def fit_func(xdata, ydata):
    try:
        K0 = ydata.iloc[-1]
        C0 = ydata.iloc[0]
        a0 = (K0 - C0)/C0
        r0 = 1
        popt, pcov = curve_fit(log_func, xdata, ydata, p0=[K0, a0, r0], bounds=([K0, 0, 0], np.inf))
        pred = log_func(xdata, *popt)
        flag = "log"
        
    except:
        try:
            popt, pcov = curve_fit(exp_func, xdata, ydata)
            pred = exp_func(xdata, *popt)
            flag = "exp"

        except:
            popt = []
            pred = []
            flag = "nope"
            
    return pred, flag, popt


def predict_country(country, country_df):
    df = conf_country_df(country_df, country, 100, True).copy()
    xdata = np.array(df.index)
    xdata += 1
    ydata = df["country"]
    
    pred, flag, popt = fit_func(xdata, ydata)
    
    # Extension
    last_indx = xdata[-1]
    extension_period = 10
    xdata_extend = np.append(xdata, np.arange(last_indx+1, last_indx+extension_period))
    
    dates = df["date"]
    init_date = df["date"].iloc[-1]
    extension = pd.date_range(init_date, periods=extension_period)[1:]
    extended_dates = pd.concat([dates, pd.Series(extension)]).reset_index(drop=True)
    
    if flag=="log":
        pred_full = log_func(xdata_extend, *popt)
        pred_full = np.round(pred_full)
    elif flag=="exp":
        pred_full = exp_func(xdata_extend, *popt)
        pred_full = np.round(pred_full)
    else:
        pred_full = []
    
    return extended_dates, pred_full, flag



# Confirmed cases
conf_raw = pd.read_sql_table('confirmed', engine, index_col=0)
confirmed = transform_export_countries_df(conf_raw)

# Deaths
death_raw = pd.read_sql_table('deaths', engine, index_col=0)
deaths = transform_export_countries_df(death_raw)

# Totals
total_confirmed = total_confirmed_df(confirmed)
total_deaths = total_confirmed_df(deaths)

world_total_confirmed = str(total_confirmed.iloc[-1, -1].astype("int"))
world_total_deaths = str(total_deaths.iloc[-1, -1].astype("int"))

# New cases
new_confirmed_cases = new_cases(confirmed)
new_confirmed_deaths = new_cases(deaths, flag_confirmed=False)

confirmed_more_100 = countries_more_than_n(confirmed, deaths, 100)
confirmed_more_1 = countries_more_than_n(confirmed, deaths, 1)

country_coords = get_country_coords(conf_raw)

new_total_cases = new_total_df(total_confirmed)
new_total_deaths = new_total_df(total_deaths)


table_confirmed = dash_table.DataTable(
	id='table_conf',
	columns=[{"name": i, "id": i} for i in new_confirmed_cases.columns],
	data=new_confirmed_cases.to_dict('records'),
	style_cell={'fontSize':14, 
				'font-family':'sans-serif',
				'textAlign': 'center',
				'overflow': 'hidden',
				'textOverflow': 'ellipsis',
				'maxWidth': 0},
	
	style_header={
		'backgroundColor': 'white',
		'fontWeight': 'bold',
		'textAlign': 'center'
	},
	style_table={
		'maxHeight': '600px',
		'overflowY': 'scroll'
	},
	style_as_list_view=True
)


table_deaths = dash_table.DataTable(
	id='table',
	columns=[{"name": i, "id": i} for i in new_confirmed_deaths.columns],
	data=new_confirmed_deaths.to_dict('records'),
	style_cell={'fontSize':14, 
				'font-family':'sans-serif',
				'textAlign': 'center',
				'overflow': 'hidden',
				'textOverflow': 'ellipsis',
				'maxWidth': 0},
	
	style_header={
		'backgroundColor': 'white',
		'fontWeight': 'bold',
		'textAlign': 'center'
	},
	style_table={
		'maxHeight': '600px',
		'overflowY': 'scroll'
	},
	style_as_list_view=True
)


def graph_totals_tab(df, new_flag, graph_id, feat_dict, layout_dict):
	if not new_flag:
		y = df["total"]
	else:
		y = df["new"]
	
	data = {'x': df["date"], 'y': y}
	data = dict(list(data.items()) + list(feat_dict.items()))
		
	return dcc.Graph(id=graph_id,
				figure={'data': [data],
						'layout': layout_dict},
				style={'height': '35vh'}
				)

		
graph1_total = graph_totals_tab(total_confirmed, False, 'graph11', dict(mode='lines+markers'), 
									dict(title="Total Confirmed Cases"))

graph1_log_total = graph_totals_tab(total_confirmed, False, 'graph11_log', dict(mode='lines+markers'), 
									{'title': "Total Confirmed Cases", 'yaxis': {'type': 'log'}})

graph1_new_total = graph_totals_tab(new_total_cases, True, 'graph11_new', dict(name="New Confirmed Cases", 
									type='bar'), dict(title="New Confirmed Cases"))

graph2_total = graph_totals_tab(total_deaths, False, "graph12", 
									dict(mode='lines+markers', marker=dict(color="Red")), dict(title="Total Deaths"))

graph2_log_total = graph_totals_tab(total_deaths, False, 'graph12_log',  
									dict(mode='lines+markers', marker=dict(color="Red")), 
									{'title': "Total Deaths", 'yaxis': {'type': 'log'}})

graph2_new_total = graph_totals_tab(new_total_deaths, True, "graph12_new", 
										dict(type='bar', marker=dict(color="Red")),
										dict(title="New Deaths"))



app.layout = html.Div([
						html.Div(
							[html.H5("COVID-19 Dashboard")],
							style={'textAlign': 'center'}
							),
						
						dcc.Tabs([
							dcc.Tab(label='Total Overview', 
									children=[
											html.Div([
													html.Div([
															html.H6("Total Confirmed Cases"),
															html.H1(world_total_confirmed)],
															style={'textAlign': 'center'},														
															className="pretty_container",
															),
													
													html.Div([html.H6("Confirmed Cases",
																style={'textAlign': 'center', 
																		'margin-top': '-1rem',
																		'margin-bottom': '0.2rem'}),
															table_confirmed],
															className="pretty_container",
															)		
													], className="three columns"),
											
											html.Div([
													html.Div([
															html.H6("Total Deaths"),
															html.H1(world_total_deaths)],
															style={'textAlign': 'center'},
															className="pretty_container",
															),
													
													html.Div([
															html.H6("Deaths",
																style={'textAlign': 'center', 
																		'margin-top': '-1rem',
																		'margin-bottom': '0.2rem'}),
															table_deaths],
															className="pretty_container",
															)	
													], className="three columns"),
													
											html.Div([
													html.Div([
															dcc.Tabs([
																	dcc.Tab(label='Linear', 
																		children=[
																			graph1_total]),
																	dcc.Tab(label='Log', 
																		children=[
																			graph1_log_total]),
																	dcc.Tab(label='New Cases', 
																		children=[
																			graph1_new_total])
																	], style={'fontSize': 12,
																				'font-family': "Arial",
																				'fontWeight': 'bold',
																				'height':'3.5vh',
																				'padding': '10px',
																				'line-height': '0.0001em'}
																	)
															], className="pretty_container"),
															
													html.Div([
															dcc.Tabs([
																	dcc.Tab(label='Linear', 
																		children=[
																			graph2_total]),
																	dcc.Tab(label='Log', 
																		children=[
																			graph2_log_total]),
																	dcc.Tab(label='New Deaths', 
																		children=[
																			graph2_new_total])
																	], style={'fontSize': 12,
																				'font-family': "Arial",
																				'fontWeight': 'bold',
																				'height':'3.5vh',
																				'padding': '10px',
																				'line-height': '0.0001em'}
																	)
															], className="pretty_container"),
															
													], className="six columns", style={'width': '54%'}),

											dcc.Markdown("""
    											[Data by John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) | \
    											[Code](https://github.com/dmanolidis/covid19-dashboard) \
    											| Developed by Dimitris Manolidis
    											""", style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'})
									
									], className="pretty_container"),
							
							dcc.Tab(label='Country Overview', 
									children=[
												
										html.Div([

												html.Div([
														dcc.Dropdown(
															id='country_sel',
															options=[{'label': i, 'value': i} for i in confirmed_more_1.columns[1:]],
															value='Italy'
															)
														],
														style={'width': '25%', 'display': 'inline-block',
														'margin-left': '1.2%', 'margin-right': '1.1%'}
														),
												]),
							
										dcc.Graph(id='graph1_country',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '35vh',
														'margin-left': '1.2%'}),
										
										dcc.Graph(id='graph2_country',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '35vh',
														'margin-right': '0.0%'}),
										
										dcc.Graph(id='graph3_country',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '35vh',
														'margin-left': '1.2%'}),
										
										dcc.Graph(id='graph4_country',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '35vh',
														'margin-right': '0.0%'}),

										html.Div([
												html.P("This tab shows an overview of the confirmed \
													cases and deaths of the selected country. Only countries \
													with at least one confirmed case and one death are shown.")],
															
													style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'},														
												),

										dcc.Markdown("""
    											[Data by John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) | \
    											[Code](https://github.com/dmanolidis/covid19-dashboard) \
    											| Developed by Dimitris Manolidis
    											""", style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'})
								
								], className="pretty_container"),
							
							dcc.Tab(label='Country Prediction', 
									children=[
												
										html.Div([

												html.Div([
														dcc.Dropdown(
															id='country_sel_prediction',
															options=[{'label': i, 'value': i} for i in confirmed_more_100.columns[1:]],
															value='Italy'
															)
														],
														style={'width': '25%', 'display': 'inline-block',
														'margin-left': '1.2%', 'margin-right': '1.1%'}
														)

												]),
							
										dcc.Graph(id='graph1_country_prediction',
												className="pretty_container",
												style={'margin-left': '1.2%', 'height': '75vh'}),
										
										html.Div([
												html.P("This tab shows a prediction for the confirmed \
													cases of the selected country. The prediction starts on the first day \
													the selected country had more than 100 total cases. The prediction \
													is based on the logistic model.")],
															
													style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'},														
												),

										dcc.Markdown("""
    											[Data by John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) | \
    											[Code](https://github.com/dmanolidis/covid19-dashboard) \
    											| Developed by Dimitris Manolidis
    											""", style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'})

								], className="pretty_container"),
							

							dcc.Tab(label='Country Comparison', 
									children=[
										
										html.Div([

												html.Div([
														dcc.Dropdown(
															id='country1',
															options=[{'label': i, 'value': i} for i in confirmed_more_100.columns[1:]],
															value='Italy'
															)
														],
														style={'width': '25%', 'display': 'inline-block',
														'margin-left': '1.2%'}),
								
												html.Div([
														dcc.Dropdown(
															id='country2',
															options=[{'label': i, 'value': i} for i in confirmed_more_100.columns[1:]],
															value='Spain'
														)
														],
														style={'width': '25%', 'display': 'inline-block'})
							
												]),
							
										dcc.Graph(id='graph1_comparison',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '75vh',
														'margin-left': '1.2%'}),
										
										dcc.Graph(id='graph2_comparison',
												className="pretty_container",
												style={'width': '46.5%', 'display': 'inline-block', 'height': '75vh'}),
										
										html.Div([
												html.P("This tab can be used to compare the data of the two selected \
													countries. The plots are aligned on the first day the countries \
													had more than 100 cases and the first day they had their first \
													death, respectively. Only countries with more than 100 cases and \
													at least one death are shown.")],
															
													style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'},														
												),

										dcc.Markdown("""
    											[Data by John Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) | \
    											[Code](https://github.com/dmanolidis/covid19-dashboard) \
    											| Developed by Dimitris Manolidis
    											""", style={'textAlign': 'center',
															'margin-left': '30%',
															'margin-right': '30%'})

									], className="pretty_container")
							
							],
							style={'fontSize': 14,
								'font-family': "Arial",
								'fontWeight': 'bold',
								'height':'7.5vh',
								'line-height': '0.5em',
								'margin-left': '0.5%',
								'margin-right': '0.8%'}
							
						)
				])


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
		
		graph1 = {'data': [dict(x=graph1_df["date"], y=graph1_df["country"],
					mode='lines+markers')],
					'layout': dict(yaxis={'type': 'log'}, title= "Confirmed Cases")}
		graph2 = {'data': [dict(x=graph2_df["date"], y=graph2_df["country"],
					mode='lines+markers', marker=dict(color="Red"))],
					'layout': dict(yaxis={'type': 'log'}, title= "Deaths")}
		graph3 = {'data': [dict(x=graph3_df["date"], y=graph3_df["new"],
					type='bar')],
					'layout': dict(title= "New Confirmed Cases")}
		graph4 = {'data': [dict(x=graph4_df["date"], y=graph4_df["new"],
					type='bar', marker=dict(color="Red"))],
					'layout': dict(title= "New Deaths")}
		
		return graph1, graph2, graph3, graph4
	
	else:
		empty_graph = {'data': [dict(x=[],y=[], mode='lines+markers')], 'layout': dict(title= "")}
		
		return empty_graph, empty_graph, empty_graph, empty_graph


@app.callback(
	Output('graph1_country_prediction', 'figure'),
	[Input('country_sel_prediction', 'value')])


def update_graph1_country_prediction(country_1):

	if country_1:
		df_selected = conf_country_df(confirmed_more_100, country_1, 100, True)
		extended_dates, pred_full, flag = predict_country(country_1, confirmed_more_100)

		data1 = dict(x=df_selected["date"], y=df_selected["country"], name=country_1, mode='lines+markers')
		if flag != "nope":
			data2 = dict(x=extended_dates, y=pred_full, name= "Prediction", mode='lines')
			data = [data2, data1]
			if flag == "log":
				title_name = country_1 + ": Logistic Prediction"
			elif flag == "exp":
				title_name = country_1 + ": Exponential Prediction"

		else:
			data = [data1]
			title_name = country_1 + " No prediction generated"
	
	else:
		data = [dict(x=[], y=[], mode='lines+markers')]
		title_name = ""
		
	return {
		'data': data,
		'layout': dict(title=title_name)}


@app.callback(
	Output('graph1_comparison', 'figure'),
	[Input('country1', 'value'),
	Input('country2', 'value')])


def update_graph1_comparison(country_1, country_2):
	if country_1 and country_2:
		c1 = conf_country_df(confirmed_more_100, country_1, 100, False)
		c2 = conf_country_df(confirmed_more_100, country_2, 100, False)
		data1 = dict(x=c1["days"], y=c1["country"], name=country_1, mode='lines+markers') 
		data2 = dict(x=c2["days"], y=c2["country"], name=country_2, mode='lines+markers')
		data = [data1, data2]
		title_name = "Confirmed Cases:   " + country_1 + " vs " + country_2
	
	elif country_1:
		c1 = conf_country_df(confirmed_more_100, country_1, 100, False)
		data1 = dict(x=c1["days"], y=c1["country"], name=country_1, mode='lines+markers')
		data = [data1]
		title_name = "Confirmed Cases:   " + country_1
	
	elif country_2:
		c2 = conf_country_df(confirmed_more_100, country_2, 100, False)
		data2 = dict(x=c2["days"], y=c2["country"], name=country_2, mode='lines+markers')
		data = [data2]
		title_name = "Confirmed Cases:   " + country_2
	
	else:
		data = [dict(x=[], y=[], mode='lines+markers')]
		title_name = ""
		
	return {
		'data': data,
		'layout': dict(yaxis={'type': 'log'},
						title=title_name)}

@app.callback(
	Output('graph2_comparison', 'figure'),
	[Input('country1', 'value'),
	Input('country2', 'value')])


def update_graph2_comparison(country_1, country_2):
	if country_1 and country_2:
		c1 = death_country_df(deaths, country_1, False)
		c2 = death_country_df(deaths, country_2, False)
		data1 = dict(x=c1["days"], y=c1["country"], name=country_1, mode='lines+markers') 
		data2 = dict(x=c2["days"], y=c2["country"], name=country_2, mode='lines+markers')
		data = [data1, data2]
		title_name = "Deaths:   " + country_1 + " vs " + country_2
	
	elif country_1:
		c1 = death_country_df(deaths, country_1, False)
		data1 = dict(x=c1["days"], y=c1["country"], name=country_1, mode='lines+markers')
		data = [data1]
		title_name = "Deaths:   " + country_1
	
	elif country_2:
		c2 = death_country_df(deaths, country_2, False)
		data2 = dict(x=c2["days"], y=c2["country"], name=country_2, mode='lines+markers')
		data = [data2]
		title_name = "Deaths:   " + country_2
	
	else:
		data = [dict(x=[], y=[], mode='lines+markers')]
		title_name = ""
		
	return {
		'data': data,
		'layout': dict(yaxis={'type': 'log'},
						title=title_name)}



if __name__ == "__main__":
	app.run_server(debug=True)