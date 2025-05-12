import plotly.graph_objects as go

# 提升值数据（示例数据需要补充完整）
categories = ['AP50↑', 'Precision↑', 'Recall↑', 'F1↑']
wtconv_z = [7.7, 9.0, 7.0, 8.0]
weyolo_g = [6.4, 3.0, 10.0, 6.5]

fig = go.Figure()

# WTConv-z
fig.add_trace(go.Scatterpolar(
    r=wtconv_z,
    theta=categories,
    fill='toself',
    name='WTConv-z',
    line=dict(color='blue')
))

# WE-YOLO-g
fig.add_trace(go.Scatterpolar(
    r=weyolo_g,
    theta=categories,
    fill='toself',
    name='WE-YOLO-g',
    line=dict(color='green', dash='dot')
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="Performance Improvement Radar Chart"
)

fig.write_image("radar_chart.png", scale=2)
fig.show()