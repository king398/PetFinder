
import plotly.express as px
import numpy
 
# Random Data
random_x = [100, 2000, 550]
names = ['A', 'B', 'C']
 
fig = px.pie(values=random_x, names=names,title="Title Here")
fig.show()