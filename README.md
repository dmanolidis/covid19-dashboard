# COVID-19 Dashboard

This dashboard monitors the development of the COVID-19 outbreak. The dashboard is live [here](https://test-dash-dman.herokuapp.com/).

The data comes from [Johns Hopkins CSSE data repository](https://github.com/CSSEGISandData/COVID-19). Furthermore, data about country death rates and population comes from [World Bank: Death Rate](https://data.worldbank.org/indicator/sp.dyn.cdrt.in) and [World Bank: Population](https://data.worldbank.org/indicator/SP.POP.TOTL) data sets, respectively.

The dashboard was created using Python and [Dash](https://dash.plotly.com/). Some of the css components are taken from this [app](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-oil-and-gas).

The app is deployed on [Heroku](https://www.heroku.com/) using free dynos, so it can probably handle only very light traffic.


### Tabs
The dashboard includes 5 tabs:

* Total Overview

    An overview of the total cases and deaths worldwide with the relevant graphs.

* Map

    This tab shows a map of the countries affected by COVID-19. The size of the bubble is proportional to the total cases of the country.

* Country Overview

    This tab shows an overview of the development of the outbreak for the selected country through graphs. 

* Country Prediction

    This tab show a prediction for the total cases for each country. The prediction is based on a simple logistic curve, as this is the expected development of an epidemic. If a logistic curve can not be fit, an exponential curve is tried next. This usually means that the country is still in the early exponential phase of the logistic curve. A confidence intervals is shown as well. The calculation of the confidence interval is based on the covariance matrix produced by `scipy.optimize.curve_fit`. This calculation is not very accurate, so any suggestions as to how to calculate the confidence interval are welcome.

* Country Comparison

    This tab shows two graphs comparing the total cases and deaths of the two selected countries. The graphs are aligned on the first day the two countries had more than 100 cases and the first day they had more than one death, respectively.

