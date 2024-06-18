import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import Counter
from plotly.subplots import make_subplots
import ast

st.set_page_config(page_title="一般特征探索", page_icon="📈",layout="wide")
file_path = 'data/Airbnb/hongkong/listings/202403_sum_listings.csv'
file_path2 = 'data/Airbnb/hongkong/listings/202403listings.csv'

neighbourhood_geojson = 'data/Airbnb/hongkong/neighbourhoods.geojson'
neighbourhood_path = 'data/Airbnb/hongkong/neighbourhoods.csv'

normal_mapbox_style="streets"
custom_mapbox_style='mapbox://styles/lithiumcoba/clwu39je101a901po19nxh814'
traffic_mapbox_style='mapbox://styles/mapbox/navigation-preview-day-v2'
mapbox_token='pk.eyJ1IjoibGl0aGl1bWNvYmEiLCJhIjoiY2x3c21hc2MzMGQ1NDJqb2J5a2Y2a2FudyJ9.h5b_gkH0cq1BV3TpvWzDqQ'
px.set_mapbox_access_token(mapbox_token)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

@st.cache_data
def add_geojson_layers(fig):
    fig.update_layout(mapbox={
        'layers': [
            {
                'source': geojson_data,
                'type': 'fill',
                'below': 'traces',
                'color': 'rgba(65,105,225,0.1)'
            },
            {
                'source': geojson_data,
                'type': 'line',
                'color': 'rgba(65,105,225,1)',
                'line': {'width': 1}
            }
        ]
    })
    
    # 添加行政区名称标签
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        name = feature['properties']['neighbourhood']  

        if feature['geometry']['type'] == 'Polygon':
            lon, lat = zip(*coordinates[0])
            centroid_lon = sum(lon) / len(lon)
            centroid_lat = sum(lat) / len(lat)
        elif feature['geometry']['type'] == 'MultiPolygon':
            lon, lat = zip(*coordinates[0][0])
            centroid_lon = sum(lon) / len(lon)
            centroid_lat = sum(lat) / len(lat)

        # 添加行政区名称标签
        fig.add_trace(go.Scattermapbox(
            lon=[centroid_lon],
            lat=[centroid_lat],
            mode='text',
            text=[name],
            showlegend=False,
            textposition='middle center',
            textfont=dict(
                size=20,
                color='dark gray',
                weight='bold'
            )
        )
        )
    return fig

with open(neighbourhood_geojson) as f:
    geojson_data = json.load(f)

df_ori=load_data(file_path)
df=df_ori
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df_detail_ori=load_data(file_path2)

df_detail=df_detail_ori[['latitude','longitude','room_type','neighbourhood_cleansed','review_scores_value','review_scores_location','review_scores_communication','review_scores_checkin','review_scores_cleanliness','review_scores_accuracy','review_scores_rating']]
df_detail = df_detail.dropna(subset=['review_scores_value', 'review_scores_location', 'review_scores_communication', 'review_scores_checkin', 'review_scores_cleanliness', 'review_scores_accuracy', 'review_scores_rating'], how='all')
df_amenity=df_detail_ori[['room_type','neighbourhood_cleansed','amenities']]

neighbourhoods=load_data(neighbourhood_path)

st.markdown("## 房源地理分布")

# 创建选择器
neighbourhoods = st.selectbox('选择行政区', ['All'] + neighbourhoods['neighbourhood'].tolist())
chart_type = st.radio(label="选择图表类型",label_visibility='hidden',options= ('散点图', '热力图'),horizontal=True)

@st.cache_data
def generate_chart(df, chart_type, traffic_mapbox_style, neighbourhoods):   
    # 根据选择的行政区筛选数据
    if neighbourhoods:
        df = df[df['neighbourhood'] == neighbourhoods]

    if chart_type == '散点图':
        fig = px.scatter_mapbox(
            df, 
            lat='latitude', 
            lon='longitude', 
            zoom=11,
            labels={'latitude':'纬度', 'longitude':'经度', 'name':'名称'},
            hover_name='name',
            color='room_type',
            mapbox_style=traffic_mapbox_style,
            title='房源分布散点图',
        )
        fig.update_layout(legend=dict(
        yanchor="top",  # y轴顶部
        y=0.99,
        xanchor="left",  # x轴靠左
        x=0.01)
        )
    elif chart_type == '热力图':
        fig = px.density_mapbox(
            df, 
            lat='latitude', 
            lon='longitude', 
            radius=10,
            zoom=11,
            color_continuous_scale='Rainbow',
            title='房源分布密度热力图',
            # mapbox_style=traffic_mapbox_style
        )
        fig.update_layout(showlegend=False)

    else:
        raise ValueError("Unsupported chart type. Please choose '散点图' or '热力图'.")
    fig.update_layout(height=800)

    return fig


map_fig = generate_chart(df, chart_type, traffic_mapbox_style, neighbourhoods if neighbourhoods != 'All' else None)
map_fig.update_layout(margin=dict(l=0, r=20, t=30, b=20))
map_fig = add_geojson_layers(map_fig)


if neighbourhoods != 'All':
    if neighbourhoods:
            df = df[df['neighbourhood'] == neighbourhoods]

bar_fig = px.bar(df, x='room_type',color='room_type', title='房源类型与价格分布直方图')
bar_fig.update_layout(showlegend=True,margin=dict(l=20, r=20, t=30, b=20),height=300)
bar_fig.update_layout(legend=dict(
        yanchor="top",  # y轴顶部
        y=0.99,
        xanchor="right",
        x=0.99)
        )
bar_fig.update_traces(hovertemplate=None)
bar_fig.update_xaxes(title_text=None,tickfont=dict(size=10))
bar_fig.update_yaxes(title_text=None)



trace1 = go.Histogram(
    x=df[df['room_type'] == 'Entire home/apt']['price'],
    opacity=0.75,
    name='Entire home/apt'
)

trace2 = go.Histogram(
    x=df[df['room_type'] == 'Private room']['price'],
    opacity=0.75,
    name='Private room'
)

trace3 = go.Histogram(
    x=df[df['room_type'] == 'Shared room']['price'],
    opacity=0.75,
    name='Shared room'
)

trace4 = go.Histogram(
    x=df[df['room_type'] == 'Hotel room']['price'],
    opacity=0.75,
    name='Hotel room'
)


layout = go.Layout(
    barmode='overlay',  # 可以改为 'stack' 实现堆叠效果
    xaxis=dict(title='Price'),
    yaxis=dict(title='Count', type='log')  # 设置y轴为对数尺度
)

# 创建图表
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),height=400)
fig.update_layout(legend=dict(
        yanchor="top",  # y轴顶部
        y=0.99,
        xanchor="right",  
        x=0.99)
        )
fig.update_xaxes(title_text='价格')
fig.update_yaxes(title_text=None)

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(map_fig, use_container_width=True)

with col2:
    with col2.container():
        st.plotly_chart(bar_fig, use_container_width=True)
    
    with col2.container():
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def create_sunburst_chart(df):
    sun_fig = px.sunburst(
        df,
        path=['neighbourhood', 'room_type'],
        title='各行政区每种房源占比',
        labels={'neighbourhood': '行政区', 'room_type': '房源类型'},
        color='neighbourhood',  # 使用行政区进行着色
        color_discrete_sequence=px.colors.qualitative.Bold  # 使用多彩的配色方案
    )

    sun_fig.update_layout(
        width=700,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        title_font=dict(size=20, family='Droid Sans Mono'),
        sunburstcolorway=["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"],
        font=dict(size=12, family='Droid Sans Mono', color='white')
    )

    # 更新标签字体
    sun_fig.update_traces(
        textfont=dict(size=14, weight='bold')
    )

    return sun_fig


sun_fig = create_sunburst_chart(df_ori)
st.plotly_chart(sun_fig,use_container_width=True)

st.divider()
st.markdown("## 房源评分特征")
chart_type2 = st.radio(label="选择分类标准",options= ('房源类型','行政区'),horizontal=True)

@st.experimental_fragment
def radar_and_histogram_charts(df, chart_type):
    #review_columns = ['review_scores_value', 'review_scores_location', 'review_scores_communication', 'review_scores_checkin', 'review_scores_cleanliness', 'review_scores_accuracy']
    review_columns = ['性价比', '位置评分', '沟通体验', '入住体验', '清洁度', '描述准确性']
    if chart_type == '行政区':
        categories = df['neighbourhood_cleansed'].unique()
        group_column = 'neighbourhood_cleansed'
    elif chart_type == '房源类型':
        categories = df['room_type'].unique()
        group_column = 'room_type'
    
    # Create subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=['细分评分雷达图', '总评分分布直方图'], specs=[[{'type': 'polar'}, {'type': 'xy'}]])

    # Add traces to the radar chart
    for category in categories:
        category_data = df[df[group_column] == category][review_columns].mean()
        fig.add_trace(go.Scatterpolar(
            r=category_data,
            theta=review_columns,
            fill='toself',
            legendgroup=category,
            name=category,
            subplot='polar'
        ), row=1, col=1)

    # Create histogram traces for different room types
    room_types = df['room_type'].unique()
    neighbourhoods = df['neighbourhood_cleansed'].unique()
    histogram_traces = []
    if chart_type == '行政区':
        for category in categories:
            histogram_traces.append(go.Histogram(
                x=df[df['neighbourhood_cleansed'] == category]['review_scores_rating'],
                opacity=0.75,
                name=category,
                legendgroup=category,                
                visible=True,
                bingroup=1
            ))
    elif chart_type == '房源类型':
        for category in categories:
            histogram_traces.append(go.Histogram(
                x=df[df['room_type'] == category]['review_scores_rating'],
                opacity=0.6,
                name=category,
                legendgroup=category,
                visible=True,
                bingroup=1
            ))

    for trace in histogram_traces:
        fig.add_trace(trace, row=1, col=2)
    
    # Update layout for interactive filtering
    fig.update_layout(        
        barmode='overlay',  # Use overlay for the histograms
        xaxis=dict(title='总评分值'),
        yaxis=dict(title=None, type='log'),  # Set y-axis to log scale
        polar=dict(
            radialaxis=dict(
                visible=True,                
                range=[4,5],                         
            ),
            ),

        height=500,  # Set the height of the figure
        margin=dict(l=20, r=20, t=20, b=0),  # Adjust the margins
        
    )
    

    return fig

df_renamed = df_detail.rename(columns={
    'review_scores_communication': '沟通体验',
    'review_scores_location': '位置评分',
    'review_scores_value': '性价比',
    'review_scores_checkin': '入住体验',
    'review_scores_cleanliness': '清洁度',
    'review_scores_accuracy': '描述准确性'
})
radar_fig = radar_and_histogram_charts(df_renamed, chart_type2)
st.plotly_chart(radar_fig,use_container_width=True)
st.divider()
st.markdown("## 房源设施特征")

def split_amenities(amenities_str):
    try:
        return ast.literal_eval(amenities_str)
    except ValueError:
        return []

@st.cache_data
def plot_amenity_heatmap(data, room_type):
    filtered_data = data
    
    all_amenities = [amenity for amenities in filtered_data['amenities'].apply(split_amenities) for amenity in amenities]

    # 统计每个amenity的出现次数
    amenity_counts = Counter(all_amenities)

    # 将统计结果转换为DataFrame
    amenity_df = pd.DataFrame.from_dict(amenity_counts, orient='index', columns=['count']).reset_index()
    amenity_df.rename(columns={'index': 'amenity'}, inplace=True)

    # 排序并选取前30的amenities
    top_30_amenities = amenity_df.sort_values(by='count', ascending=False).head(30)

    if room_type == '展示全部房型情况':
        heatmap_data = top_30_amenities[['count']].T
        heatmap_data.columns = top_30_amenities['amenity']
        heatmap_y = ['Count']
    else:
        # 按房间类型计算每个amenity的比例
        room_type_groups = filtered_data.groupby('room_type')
        room_type_amenity_counts = {
            room: Counter([amenity for amenities in group['amenities'].apply(split_amenities) for amenity in amenities])
            for room, group in room_type_groups
        }

        heatmap_data = pd.DataFrame({
            room_type: {amenity: room_type_amenity_counts[room_type].get(amenity, 0) / len(filtered_data[filtered_data['room_type'] == room_type])
                        for amenity in top_30_amenities['amenity']}
            for room_type in room_type_groups.groups.keys()
        }).T
        heatmap_y = heatmap_data.index

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_y,
        colorscale='YlGnBu',
        showscale=True,
        texttemplate="%{text}",
        textfont={"size":10}
    ))

    if room_type == '展示全部房型情况':
        fig.update_layout(
            title=f'全部房型房间设施热力图',
            title_font=dict(size=20),            
            xaxis_nticks=36,
            height=600,
        )
        fig.update_yaxes(visible=False)
    else:
        fig.update_layout(
            title=f'各房型房间设施热力图',
            title_font=dict(size=20),
            xaxis_nticks=36,
            height=600,
        )
    fig.update_xaxes(tickfont=dict(size=15))

    st.plotly_chart(fig, use_container_width=True)

room_types = ['展示全部房型情况', '分房型展示']
selected_room_type = st.radio('选择展示方式:', room_types, horizontal=True)

plot_amenity_heatmap(df_amenity, selected_room_type)
st.divider()
st.markdown('作者：同济大学 信息管理与信息系统专业 李佳佳')