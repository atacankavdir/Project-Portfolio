import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import recommendation_function as rf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
                 [
                        html.H6(
                            "Welcome to the Product Recommendation Engine",
                            style = {'color' : 'white'}
                        ),
                     
                        html.Div(
                            [
                                "ProductId   : ", 
                                dcc.Input(
                                    id='input-1-state',
                                    type='text',
                                    value = '23129, 127384, 94234, 80874'
                                )
                            ]),
                        html.Div(
                            [
                                "CustomerId  : ", 
                                dcc.Input(id='input-2-state', 
                                          type='text',  
                                          value = None
                                )
                            ]
                        ),
                     
                        html.Button(
                            id='submit-button-state', 
                            n_clicks=0, 
                            children='Submit',
                            style = {'background-color' : 'white'}
                        ),
                     
                        html.Div(
                            id='output-state',
                            style = {'color' : 'white'}
                        )
                     
                 ],  
    style={'width': '80%',
           'padding-left':'40%', 
           'padding-right':'50%' , 
           'background-color':'lightgrey'})


@app.callback(Output('output-state', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('input-1-state', 'value'),
              State('input-2-state', 'value'))

def update_output(n_clicks, input1, input2):
    
    output_value = list(input1.replace(" ", "").split(","))

    numbers = [int(x) for x in output_value]
    
    recom_list = recommendation_engine(numbers, 
                                       input2,
                                       rf.purchases,
                                       rf.views,
                                       rf.products
                                       )
        
    return "Recommended order of the products are: "+ str(recom_list)


if __name__ == '__main__':
    app.run_server(debug=False)
