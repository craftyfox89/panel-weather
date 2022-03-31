import panel as pn
import pandas as pd
import xarray as xr
import holoviews as hv
from bokeh.models import HoverTool, FixedTicker
import numpy as np
import matplotlib.colors as mcolors
import panel.widgets as pnw
import holoviews.operation.datashader as hd

hv.extension('bokeh')
pn.extension(sizing_mode='stretch_width')
pn.extension()

xlm = (680000, 1150000)
ylm = (5750000, 6070000)
transparency = 'B3'

css = '''
.bk.style-cards {
  font-family: 'Courier New'
}
'''

pn.extension(raw_css=[css])

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def generateCmap(dat, param):
    if param == 'temperature' or param == 'dewpoint' or param == 'thetaE':
        mn = -30
        mx = 54
        step = 2.
        cmapValue = np.arange(mn, mx, step)
        cmapDiscrete = np.array(['#818281', '#CCCCCD', '#FFFFFF', '#F9E7FD', '#F2CBFB', '#E472F6', '#CE38DB', '#A66ABB',
                    '#8023A9', '#031290', '#051BC0', '#2146F3', '#4A8DE9', '#61B4F7', '#A7E5FB', '#3D7520',
                    '#549D42', '#6AC33B', '#88E274', '#A8FB98', '#C4FCBC', '#FBFAAB', '#F9F486', '#F2DC80',
                    '#EEB54A', '#E38F38', '#CC5926', '#A6341A', '#80170f', '#a0110b', '#c00c08', '#df0604', '#ff0000',
                    '#fe3c3c', '#fd7878', '#fcb4b4', '#fbf0f0', '#FFFFFF', '#CCCCCD', '#818281', '#000000'])
    if param == 'pressure':
        mn = 500
        mx = 1100
        step = 50.
        cmapValue = np.arange(mn, mx, step)
        cm = mcolors.LinearSegmentedColormap.from_list("Custom", ['darkcyan','whitesmoke','tab:brown'], N=len(cmapValue)-1)
        cmapDiscrete = [mcolors.rgb2hex(cm(i)) for i in range(cm.N)]

    fst = np.nanmin(dat)
    lst = np.nanmax(dat)
    lim_array = np.arange(np.ceil(fst / step) * step, step + np.floor(lst / step) * step, step)
    lim = NormalizeData(np.array([fst] + lim_array.tolist() + [lst]))
    limRnd = (lim*1000).round()

    st = np.where(cmapValue > fst)[0][0]-1
    et = np.where(cmapValue > lst)[0][0]-1
    cmapD = cmapDiscrete[st:et+1]

    ix1 = limRnd[0:len(limRnd)-1]
    ix2 = limRnd[1:len(limRnd)]
    ix = ix2 - ix1

    cmap = []
    for ii, col in enumerate(cmapD):
        cmap.extend(np.repeat(col + str(transparency), ix[ii]))

    return cmap, cmapValue


def plot(dat, df, param):
    colormapPoints, tickList = generateCmap(df[param], dat.attrs['parameter'])
    colormapImage, tickList = generateCmap(dat.values, dat.attrs['parameter'])
    image_height, image_width = 600, 1020
    map_height, map_width = image_height, 1020
    key_dimensions = ['x', 'y']
    value_dimension = param

    tooltips1 = [
        ('', '@image'),
    ]
    hover1 = HoverTool(tooltips=tooltips1)
    tooltips2 = [
        ('', '@' + param),
    ]
    hover2 = HoverTool(tooltips=tooltips2)

    hv.extension('bokeh', logo=False)
    hv_tiles_osm = hv.element.tiles.StamenToner()
    clipping = {'NaN': '#00000000'}
    hv.opts.defaults(
      hv.opts.Points(cmap=colormapPoints, line_color='black', size=8, tools=[hover2], padding=0.05),
      hv.opts.Image(cmap=colormapImage, height=image_height, width=image_width,
                    colorbar=True, tools=[hover1], active_tools=['wheel_zoom'],
                    clipping_colors=clipping, xlim=xlm, ylim=ylm),
      hv.opts.Tiles(active_tools=['wheel_zoom'], height=map_height, width=map_width,
                    xlim=xlm, ylim=ylm)
    )
    hv_dataset_large = hv.Dataset(dat, vdims=value_dimension, kdims=key_dimensions)
    min_val = float(hv_dataset_large.data[value_dimension].min())
    max_val = float(hv_dataset_large.data[value_dimension].max())
    hv_image_large = hv.Image(hv_dataset_large).opts(colorbar_opts={'ticker': FixedTicker(ticks=tickList)})
    hv_points = hv.Points(data=df, kdims=['x', 'y'], vdims=[param]).opts(color=param)
    hv_dyn_large = hd.regrid(hv_image_large).opts(clim=(min_val, max_val))
    fig = hv_tiles_osm * hv_dyn_large * hv_points
    return fig


def load_data(variable='Température [°C]', view_fn=plot):
    if variable == 'Température [°C]':
        dat = xr.open_dataarray("database/temperature.nc")
        df = pd.read_csv("database/temperature.csv")
        param = 'temperature'
    if variable == 'Point de rosée [°C]':
        dat = xr.open_dataarray("database/dewpoint.nc")
        df = pd.read_csv("database/dewpoint.csv")
        param = 'dewpoint'
    if variable == 'Pression [hPa]':
        dat = xr.open_dataarray("database/pressure.nc")
        df = pd.read_csv("database/pressure.csv")
        param = 'pressure'
    if variable == 'ThetaE [°C]':
        dat = xr.open_dataarray("database/thetaE.nc")
        df = pd.read_csv("database/thetaE.csv")
        param = 'thetaE'
    return view_fn(dat, df, param)


def load_scores(variable='Température [°C]'):
    if variable == 'Température [°C]':
        scores = pd.read_csv('database/scores_temperature.csv')
    if variable == 'Point de rosée [°C]':
        scores = pd.read_csv('database/scores_dewpoint.csv')
    if variable == 'Pression [hPa]':
        scores = pd.read_csv('database/scores_pressure.csv')
    if variable == 'ThetaE [°C]':
        scores = pd.read_csv('database/scores_thetaE.csv')
    df_pane = pn.widgets.DataFrame(scores, show_index=False,
                                   autosize_mode='fit_viewport',
                                   row_height=20)
    return df_pane


def load_scores_hist(variable='Température [°C]'):
    if variable == 'Température [°C]':
        scores = pd.read_csv('database/scoresH_temperature.csv', parse_dates=['time'])
    if variable == 'Point de rosée [°C]':
        scores = pd.read_csv('database/scoresH_dewpoint.csv', parse_dates=['time'])
    if variable == 'Pression [hPa]':
        scores = pd.read_csv('database/scoresH_pressure.csv', parse_dates=['time'])
    if variable == 'ThetaE [°C]':
        scores = pd.read_csv('database/scoresH_thetaE.csv', parse_dates=['time'])

    scores = scores[scores.time > scores.time.iloc[-1] + pd.DateOffset(-1)]

    if variable == 'ThetaE [°C]':
        plt_pane = hv.Curve(
            data=scores,
            kdims=['time'],
            vdims=['num sta'],
        )
    else:
        plt_pane = hv.Curve(
            data=scores,
            kdims=['time'],
            vdims=['trend r2'],
        )
    tooltips = [
        ('r2', '$y'),
    ]
    hover = HoverTool(tooltips=tooltips, mode='vline')
    plt_pane = plt_pane.opts(tools=[hover], height=150, toolbar=None,
                             default_tools=[])
    return plt_pane


def load_time(variable='Température [°C]'):
    curTime = pd.read_csv("database/curTime.csv")['Current Time'][0][11:16]
    text = "####" + variable + " - " + curTime + "<hr>"
    return pn.pane.Markdown(text, style={'height': '50px', 'font-family': 'Courier New'})


variable = pnw.RadioBoxGroup(name='variable', value='Température [°C]',
                             options=['Température [°C]',
                                      'Point de rosée [°C]',
                                      'Pression [hPa]',
                                      'ThetaE [°C]'])

retrieve_time = pn.bind(load_time, variable)
retrieve_data = pn.bind(load_data, variable)
retrieve_scores = pn.bind(load_scores, variable)
retrieve_scores_hist = pn.bind(load_scores_hist, variable)

bootstrap = pn.template.BootstrapTemplate(title='Valeurs actuelles')
bootstrap.sidebar.append(
    pn.Row(
        pn.Card(variable, title='Paramètres',
                css_classes=['style-cards']),
    )
)
bootstrap.sidebar.append(
    pn.Row(
        pn.Card(retrieve_scores, title='Scores',
                css_classes=['style-cards']),
    )
)
bootstrap.sidebar.append(
    pn.Row(
        pn.Card(retrieve_scores_hist, title='Historique des scores',
                css_classes=['style-cards']),
    )
)
bootstrap.main.append(
    pn.Column(retrieve_time, retrieve_data)
)


bootstrap.servable()
