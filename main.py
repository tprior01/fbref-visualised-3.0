from pandas import read_csv
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import LabelSet, ColumnDataSource, Div, Select, AutocompleteInput, RangeSlider, DataRange1d, Circle
from numpy import linspace, abs, where, array, abs as nabs, median
from bokeh.plotting import figure, show
from scipy.stats import lognorm
from maps import *
from json import load
from adjustText import adjust_text
from matplotlib.pyplot import plot, text
from functools import partial

# lookups
with open('data/lists/names.json') as json_file:
    name_list = load(json_file)
with open('data/lists/teams.json') as json_file:
    team_list = load(json_file)
with open('data/lists/nations.json') as json_file:
    nation_list = load(json_file)
with open('data/dictionaries/ids.json') as json_file:
    id_dict = load(json_file)
with open('data/dictionaries/positions.json') as json_file:
    position_dict = load(json_file)
with open('data/dictionaries/names.json') as json_file:
    name_dict = load(json_file)
with open('data/constants/max_min.json') as json_file:
    max_min = load(json_file)
with open('data/constants/max_age.json') as json_file:
    max_age = load(json_file)
with open('data/constants/max_value.json') as json_file:
    max_value = load(json_file)

# text to display at the top of page
desc = Div(text=open(f'description.html').read(), sizing_mode="stretch_width")

# data load
df = read_csv('data/data.csv', header=[0, 1], index_col=0, dtype=data_types)
pc = read_csv('data/percentiles.csv', header=[0, 1, 2], index_col=0, dtype=percentile_data_types).fillna(0)
op = read_csv('data/axis_options.csv', index_col=0)

# toggles
total_per_min = Select(title='Total or Per Minute', value='Per Minute', options=['Total', 'Per Minute'])
distribution_axis = Select(title='Distribution Plot:', value='Y-Axis', options=['X-Axis', 'Y-Axis'])

# axis options
x_category = Select(title='X-Axis Category', value='Passing', options=list(category_map.keys()))
y_category = Select(title='Y-Axis Category', value='Defensive Actions', options=list(category_map.keys()))
x_axis = Select(title='X-Axis', value='Progressive Passes Completed', options=list(op[x_category.value].dropna()))
y_axis = Select(title='Y-Axis', value='Tackles + Interceptions', options=list(op[y_category.value].dropna()))

# range sliders
minutes = RangeSlider(title='Number of minutes', value=(1350, max_min), start=0, end=max_min, step=10)
value = RangeSlider(title='Value', value=(0, max_value), start=0, end=max_value, step=1)
age = RangeSlider(title='Age', value=(15, max_age), start=15, end=max_age, step=1)

# select options
historic_team = Select(title='Historic Team:', value='All', options=team_list)
current_team = Select(title='Current Team:', value='All', options=team_list)
nation = Select(title='Nation:', value='All', options=nation_list)
show_text = Select(title='Player Name Labels:', value="Don't Show", options=['Show', "Don't Show"])


# highlight player options
name = AutocompleteInput(title='Highlighted player name:', value='Takehiro Tomiyasu', completions=name_list,
                         restrict=True, case_sensitive=False)
id_list = Select(title='Highlighted player ID:', value='b3af9be1', options=id_dict[name.value])

# column data sources
scatter_data = dict(
    GK=ColumnDataSource(data=dict(x=[], y=[])),
    CB=ColumnDataSource(data=dict(x=[], y=[])),
    FB=ColumnDataSource(data=dict(x=[], y=[])),
    DM=ColumnDataSource(data=dict(x=[], y=[])),
    CM=ColumnDataSource(data=dict(x=[], y=[])),
    AM=ColumnDataSource(data=dict(x=[], y=[])),
    WI=ColumnDataSource(data=dict(x=[], y=[])),
    FW=ColumnDataSource(data=dict(x=[], y=[]))
)
scatter_highlighted = ColumnDataSource(data=dict(x=[], y=[]))
distribution_data = dict(
    ALL=ColumnDataSource(data=dict(x=[], y=[])),
    GK=ColumnDataSource(data=dict(x=[], y=[])),
    CB=ColumnDataSource(data=dict(x=[], y=[])),
    FB=ColumnDataSource(data=dict(x=[], y=[])),
    DM=ColumnDataSource(data=dict(x=[], y=[])),
    CM=ColumnDataSource(data=dict(x=[], y=[])),
    AM=ColumnDataSource(data=dict(x=[], y=[])),
    WI=ColumnDataSource(data=dict(x=[], y=[])),
    FW=ColumnDataSource(data=dict(x=[], y=[]))
)
distribution_highlighted = dict(
    ALL=ColumnDataSource(data=dict(x=[], y=[])),
    GK=ColumnDataSource(data=dict(x=[], y=[])),
    CB=ColumnDataSource(data=dict(x=[], y=[])),
    FB=ColumnDataSource(data=dict(x=[], y=[])),
    DM=ColumnDataSource(data=dict(x=[], y=[])),
    CM=ColumnDataSource(data=dict(x=[], y=[])),
    AM=ColumnDataSource(data=dict(x=[], y=[])),
    WI=ColumnDataSource(data=dict(x=[], y=[])),
    FW=ColumnDataSource(data=dict(x=[], y=[]))
)
text_data = ColumnDataSource(data=dict(x=[], y=[]))

# tooltips
tooltips = [
    ('Name', '@name'),
    ('Position', '@Pos'),
    ('Age', '@Age'),
    ('Team', '@Team'),
    ('Nationality', '@Nation'),
    ('Value', '€@{Value}M'),
    ('X-Axis', '@x{0.000}'),
    ('Y-Axis', '@y{0.000}')
]

# renderer lists and dictionaries
s_position_renderers = []
s_position_renderers_dictionary = dict()
s_text_renderer_dictionary = dict()

# scatter plot
s = figure(tools='tap', height=700, width=700, title='', tooltips=tooltips,
           sizing_mode='scale_both', toolbar_location=None)
for position, data, colour in zip(scatter_data.keys(), scatter_data.values(), colours):
    temp = s.circle(
        x='x', y='y', source=scatter_data[position], size=5, color=colour, line_color=None, legend_label=position
    )
    s_position_renderers.append(temp)
    s_position_renderers_dictionary[position] = temp
    labels = LabelSet(x='x', y='y', text='short', source=scatter_data[position], text_font_size="7pt",
                      text_align='text_align', text_baseline='text_baseline')
    s.add_layout(labels)
    s_text_renderer_dictionary[position] = labels
s.legend.click_policy = "hide"
s.x_range = DataRange1d(only_visible=True, renderers=s_position_renderers)
s.y_range = DataRange1d(only_visible=True, renderers=s_position_renderers)
s.hover.renderers = s_position_renderers
for renderer in s_position_renderers:
    renderer.nonselection_glyph = None
glyph = Circle(fill_alpha=1, fill_color="yellow", line_color='black', line_width=2)
s.add_glyph(scatter_highlighted, glyph)
s.add_layout(s.legend[0], 'right')
s.legend.spacing = 1

# distribution plot
d_legend_items = dict()
d_renderers = []
d_glyph_items = dict()
d = figure(title='', height=150, toolbar_location=None)
for position, colour in zip(distributions, distribution_colours):
    temp = d.line(x='x', y='y', source=distribution_data[position], line_color=colour, line_width=2, alpha=0.7,
                  legend_label=position)
    d_renderers.append(temp)
    d_legend_items[position] = temp
    temp = d.circle(x='x', y='y', source=distribution_highlighted[position], size=7, fill_alpha=1, fill_color="yellow",
                    line_color=colour, line_width=2)
    d_glyph_items[position] = temp
d.legend.click_policy = 'hide'
d.yaxis.axis_label = 'Pr(x)'
d.y_range = DataRange1d(only_visible=True, renderers=d_renderers)
d.add_layout(d.legend[0], 'right')
d.legend.spacing = 1


def select_players():
    selected = df[
        (df.loc[:, ('stats', 'Min')] >= minutes.value[0]) &
        (df.loc[:, ('stats', 'Min')] <= minutes.value[1]) &
        (df.loc[:, ('stats', 'Age')] >= age.value[0]) &
        (df.loc[:, ('stats', 'Age')] <= age.value[1]) &
        (df.loc[:, ('stats', 'Value')] >= value.value[0]) &
        (df.loc[:, ('stats', 'Value')] <= value.value[1])
        ]
    if historic_team.value != 'All':
        selected = selected[selected.loc[:, ('stats', 'Squad')].str.contains(historic_team.value)]
    if current_team.value != 'All':
        selected = selected[selected.loc[:, ('stats', 'Team')] == current_team.value]
    if nation.value != 'All':
        selected = selected[selected.loc[:, ('stats', 'Nation')] == nation.value]
    return selected


def update_scatter():
    selected = select_players()
    y_cat = category_map[y_category.value]
    y_sub_cat = axis_maps[y_cat][y_axis.value]
    x_cat = category_map[x_category.value]
    x_sub_cat = axis_maps[x_cat][x_axis.value]
    not_per_min_x = x_sub_cat not in not_per_min
    not_per_min_y = y_sub_cat not in not_per_min
    if total_per_min.value == 'Per Minute':
        s.xaxis.axis_label = f'{x_axis.value} / Minute' if not_per_min_x else x_axis.value
        s.yaxis.axis_label = f'{y_axis.value} / Minute' if not_per_min_y else y_axis.value
    else:
        s.xaxis.axis_label = x_axis.value
        s.yaxis.axis_label = y_axis.value
    for position, data in scatter_data.items():
        df = selected[selected.loc[:, ('stats', 'TmPos')] == position_map[position]]
        if total_per_min.value == 'Per Minute':
            x = (df.loc[:, (x_cat, x_sub_cat)] / df.loc[:, ('stats', 'Min')]) if not_per_min_x else df.loc[:,
                                                                                                    (x_cat, x_sub_cat)]
            y = (df.loc[:, (y_cat, y_sub_cat)] / df.loc[:, ('stats', 'Min')]) if not_per_min_y else df.loc[:,
                                                                                                    (y_cat, y_sub_cat)]
        else:
            x = df.loc[:, (x_cat, x_sub_cat)]
            y = df.loc[:, (y_cat, y_sub_cat)]
        dictionary = dict(x=x,
                          y=y,
                          Pos=df.loc[:, ('stats', 'TmPos')],
                          name=df.loc[:, ('stats', 'Player')],
                          Nation=df.loc[:, ('stats', 'Nation')],
                          Age=df.loc[:, ('stats', 'Age')],
                          Team=df.loc[:, ('stats', 'Team')],
                          Value=df.loc[:, ('stats', 'Value')],
                          id=df.index,
                          short=df.loc[:, ('stats', 'short')],
                          text_align=['right' for x in range(len(x))],
                          text_baseline=['right' for x in range(len(x))]
                          )
        data.data = dictionary


def find_closest(arr, val):
    idx = abs(arr - val).argmin()
    return arr[idx]


def reject_outliers(data, m=9.):
    d = nabs(data - median(data))
    mdev = median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def update_distributions():
    if distribution_axis.value == 'Y-Axis':
        title = y_axis.value
        cat = category_map[y_category.value]
        sub_cat = axis_maps[cat][title]
    else:
        title = x_axis.value
        cat = category_map[x_category.value]
        sub_cat = axis_maps[cat][title]
    not_per_min_axis = sub_cat not in not_per_min
    max_pdfs = []
    for distribution in distributions:
        if distribution == 'ALL':
            condition = (df.loc[:, ('stats', 'Min')] >= 1350)
        else:
            condition = (df.loc[:, ('stats', 'Min')] >= 1350) & (df.loc[:, ('stats', 'TmPos')] == position_map[distribution])
        values = df.loc[:, (cat, sub_cat)][condition]
        if not_per_min_axis:
            minutes = df.loc[:, ('stats', 'Min')][condition]
            values = values / minutes
        x = linspace(values.min(), values.max(), 1001)
        params = lognorm.fit(values)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        pdf = lognorm.pdf(x, loc=loc, scale=scale, *arg)
        distribution_data[distribution].data = dict(x=x, y=pdf)
        d_legend_items[distribution].visible = False
        max_pdfs.append(max(pdf))
    if cat == 'keepers' or cat == 'keepersadv':
        for position in distributions:
            if position == 'GK':
                d_legend_items[position].visible = True
            else:
                d_legend_items[position].visible = False
    else:
        max_pdfs = array(max_pdfs)
        okay_pdfs = reject_outliers(reject_outliers(max_pdfs))
        for i in range(9):
            if max_pdfs[i] in okay_pdfs:
                d_legend_items[distributions[i]].visible = True
            else:
                d_legend_items[distributions[i]].visible = False
    if not_per_min_axis:
        d.xaxis.axis_label = f'x: {title} / Minute'
    else:
        d.xaxis.axis_label = f'x: {title}'


def update_text():
    if show_text.value == "Don't Show":
        for position in positions:
            s_text_renderer_dictionary[position].visible = False
    else:
        x = []
        y = []
        shorts = []
        lengths = dict()
        for position, data in scatter_data.items():
            if not s_position_renderers_dictionary[position].visible:
                s_text_renderer_dictionary[position].visible = False
            else:
                s_text_renderer_dictionary[position].visible = True
                l = len(data.data['x'])
                lengths[position] = l
                x.extend(data.data['x'])
                y.extend(data.data['y'])
                shorts.extend(data.data['short'])
        l = sum(lengths.values())
        if l > 75:
            return
        plot(x, y)
        texts = [text(x[i], y[i], shorts[i], ha='center', va='center', size=7) for i in range(l)]
        adjust_text(texts)
        ha = [item.get_ha() for item in texts]
        va = [item.get_va() for item in texts]
        i = 0
        for position, data, length in zip(scatter_data.keys(), scatter_data.values(), lengths.values()):
            data.data['text_align'] = ha[i: i + length]
            data.data['text_baseline'] = va[i: i + length]
            i += length


def update_size():
    size = 0
    df = select_players()
    for position in positions:
        if s_position_renderers_dictionary[position].visible:
            size += len(df[df.stats.TmPos == position_map[position]])
    s.title.text = f'{size} players selected, seasons 2017-2018 to 2022-23'


def ords(n):
    if n == 0:
        return '1st'
    elif n == 100:
        return '99th'
    else:
        return str(n) + ("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def update_highlighted():
    y_cat = category_map[y_category.value]
    y_sub_cat = axis_maps[y_cat][y_axis.value]
    x_cat = category_map[x_category.value]
    x_sub_cat = axis_maps[x_cat][x_axis.value]
    not_per_min_x = x_sub_cat not in not_per_min
    not_per_min_y = y_sub_cat not in not_per_min
    loc = df.loc[id_list.value]
    if total_per_min.value == 'Per Minute':
        x = [(loc[x_cat][x_sub_cat] / loc.stats.Min) if not_per_min_x else loc[x_cat][x_sub_cat]]
        y = [(loc[y_cat][y_sub_cat] / loc.stats.Min) if not_per_min_y else loc[y_cat][y_sub_cat]]
    else:
        x = [loc[x_cat][x_sub_cat]]
        y = [loc[y_cat][y_sub_cat]]
    scatter_highlighted.data = dict(
        x=x,
        y=y
    )
    if distribution_axis.value == 'Y-Axis':
        loc = df.loc[id_list.value]
        cat = y_cat
        data = loc[cat][y_sub_cat] / loc.stats.Min if not_per_min_y else loc[y_cat][y_sub_cat]
        perc = pc.loc[id_list.value, ('perc', cat, y_sub_cat)]
        posp = pc.loc[id_list.value, ('posp', cat, y_sub_cat)]
    else:
        loc = df.loc[id_list.value]
        cat = x_cat
        data = loc[cat][x_sub_cat] / loc.stats.Min if not_per_min_x else loc[x_cat][x_sub_cat]
        perc = pc.loc[id_list.value, ('perc', cat, x_sub_cat)]
        posp = pc.loc[id_list.value, ('posp', cat, x_sub_cat)]
    for distribution in distributions:
        if loc.stats.TmPos == position_map[distribution] or distribution == 'ALL':
            index = where(
                distribution_data[distribution].data['x'] == find_closest(distribution_data[distribution].data['x'],
                                                                          data))[0][0]
            distribution_highlighted[distribution].data = dict(
                x=[distribution_data[distribution].data['x'][index]],
                y=[distribution_data[distribution].data['y'][index]]
            )
            if loc.stats.Min <= 1350:
                d.title.text = f"{loc.stats.Player} has played less than 1350 minutes"
            elif 'keepers' in loc[cat] and loc.keepers.GkMin <= 1350:
                d.title.text = f"{loc.stats.Player} has played less than 1350 minutes in goal"
            else:
                d.title.text = f"{loc.stats.Player} is in the {ords(perc)} percentile overall and the {ords(posp)} percentile for {position_map[distribution].lower()}s"
        else:
            distribution_highlighted[distribution].data = dict(
                x=[],
                y=[]
            )


def update_id():
    id_list.options = id_dict[name.value]
    if name.value != name_dict[id_list.value]:
        id_list.value = id_list.options[0]


def update_x_options(attr, old, new):
    if new:
        options = list(op[new].dropna())
        x_axis.options = options
        x_axis.value = options[0]


def update_y_options(attr, old, new):
    if new:
        options = list(op[new].dropna())
        y_axis.options = options
        y_axis.value = options[0]


def updated_highlighted_dist():
    for position in positions:
        if d_legend_items[position].visible:
            d_glyph_items[position].visible = True
        else:
            d_glyph_items[position].visible = False


def selection_callback(attr, old, new, position):
    try:
        id_list.value = scatter_data[position].data['id'][new[0]]
        name.value = name_dict[id_list.value]
    except IndexError:
        pass


def distribution_callback(attr, old, new, position):
    if d_legend_items[position].visible:
        d_glyph_items[position].visible = True
    else:
        d_glyph_items[position].visible = False


def text_callback(attr, old, new, position):
    if show_text.value == 'Show':
        if s_position_renderers_dictionary[position].visible:
            s_text_renderer_dictionary[position].visible = True
        else:
            s_text_renderer_dictionary[position].visible = False
    else:
        s_text_renderer_dictionary[position].visible = False


# update categories events
x_category.on_change('value', update_x_options)
y_category.on_change('value', update_y_options)

# update scatter events
x_axis.on_change('value', lambda attr, old, new: update_scatter())
y_axis.on_change('value', lambda attr, old, new: update_scatter())
historic_team.on_change('value', lambda attr, old, new: update_scatter())
current_team.on_change('value', lambda attr, old, new: update_scatter())
nation.on_change('value', lambda attr, old, new: update_scatter())
age.on_change('value', lambda attr, old, new: update_scatter())
value.on_change('value', lambda attr, old, new: update_scatter())
minutes.on_change('value', lambda attr, old, new: update_scatter())
total_per_min.on_change('value', lambda attr, old, new: update_scatter())

# update distribution events
x_axis.on_change('value', lambda attr, old, new: update_distributions())
y_axis.on_change('value', lambda attr, old, new: update_distributions())
distribution_axis.on_change('value', lambda attr, old, new: update_distributions())

# update highlighted events
x_axis.on_change('value', lambda attr, old, new: update_highlighted())
y_axis.on_change('value', lambda attr, old, new: update_highlighted())
id_list.on_change('value', lambda attr, old, new: update_highlighted())

# update size (in title) events
minutes.on_change('value', lambda attr, old, new: update_size())
historic_team.on_change('value', lambda attr, old, new: update_size())
current_team.on_change('value', lambda attr, old, new: update_size())
nation.on_change('value', lambda attr, old, new: update_size())
age.on_change('value', lambda attr, old, new: update_size())
value.on_change('value', lambda attr, old, new: update_size())
for position in positions:
    s_position_renderers_dictionary[position].on_change('visible', lambda attr, old, new: update_size())

# update id events
name.on_change('value', lambda attr, old, new: update_id())

selection_renderer_dict = {0: 'GK', 1: 'CB', 2: 'FB', 3: 'DM', 4: 'CM', 5: 'AM', 6: 'WI', 7: 'FW'}
for i in range(8):
    s_position_renderers[i].data_source.selected.on_change('indices', partial(selection_callback, position=selection_renderer_dict[i]))

# update highlight player events for distribution plots
distribution_axis.on_change('value', lambda attr, old, new: updated_highlighted_dist())
distribution_renderer_dict = {0: 'ALL', 1: 'GK', 2: 'CB', 3: 'FB', 4: 'DM', 5: 'CM', 6: 'AM', 7: 'WI', 8: 'FW'}
for i in range(9):
    s_position_renderers[i].data_source.selected.on_change('indices', partial(distribution_callback, position=distribution_renderer_dict[i]))


# text callback
show_text.on_change('value', lambda attr, old, new: update_text())
for i in range(8):
    s_position_renderers[i].on_change('visible', partial(text_callback, position=selection_renderer_dict[i]))


# controls and layout
controls = [total_per_min, minutes, age, value, x_category, x_axis, y_category, y_axis, historic_team, current_team, nation,
            name, id_list, distribution_axis, show_text]
inputs = column(*controls, width=245, spacing=-8)
layout = column(desc, row(inputs, s), d, sizing_mode='scale_both', max_width=1300)

update_scatter()
update_size()
update_distributions()
update_highlighted()
update_text()

curdoc().add_root(layout)
curdoc().title = 'FBRef Visualised'
show(layout)