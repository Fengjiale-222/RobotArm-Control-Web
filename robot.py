import streamlit as st
import numpy as np
import plotly.graph_objects as go
import socket
import json
import paho.mqtt.client as mqtt
import time

# ==========================================
# 1. 机械臂物理参数 (数字孪生模型)
# ==========================================
class RobotParams:
    L1 = 0.08; L2 = 0.155; L3 = 0.11; L4 = 0.11
    d2 = -0.0165; d3 = -0.0165
p = RobotParams()

# ==========================================
# 2. 网页端仿真算法 (碰撞拦截)
# ==========================================
def calculate_4dof_ik(x, y, z):
    try:
        r_dist = np.sqrt(x**2 + y**2)
        if r_dist < 0.005: return None, "靠近中心死区"
        j1_math = np.arctan2(y, x)
        r_j4 = r_dist; z_j4 = z + p.L4 - p.L1
        
        A = np.sqrt(p.L2**2 + p.d2**2); alpha1 = np.arctan2(p.d2, p.L2)
        B = np.sqrt(p.L3**2 + p.d3**2); alpha2 = np.arctan2(p.d3, p.L3)
        D_sq = r_j4**2 + z_j4**2; D = np.sqrt(D_sq)
        if D > (A + B) or D < abs(A - B): return None, "目标超出臂展范围"
            
        cos_u2 = np.clip((D_sq - A**2 - B**2) / (2 * A * B), -1, 1)
        u2 = -np.arccos(cos_u2) 
        u1 = np.arctan2(z_j4, r_j4) - np.arctan2(B * np.sin(u2), A + B * np.cos(u2))
        
        m1, m2, m3 = j1_math, u1 - alpha1, u2 + alpha1 - alpha2
        m4 = -np.pi/2 - m2 - m3
        return np.array([np.degrees(m1), np.degrees(np.pi/2 - m2), np.degrees(m3 + np.pi/2), np.degrees(-m4)]), "校验通过"
    except Exception as e:
        return None, f"解算错误: {str(e)}"

def forward_kinematics_3d(hw_angles):
    j1, j2, j3, j4 = np.radians(hw_angles)
    m1, m2, m3 = j1, np.pi/2 - j2, j3 - np.pi/2
    pts = [[0, 0, 0], [0, 0, p.L1]] 
    x3 = np.cos(m1) * (p.L2 * np.cos(m2) - p.d2 * np.sin(m2))
    y3 = np.sin(m1) * (p.L2 * np.cos(m2) - p.d2 * np.sin(m2))
    z3 = p.L1 + p.L2 * np.sin(m2) + p.d2 * np.cos(m2)
    pts.append([x3, y3, z3])
    m23 = m2 + m3
    x4 = x3 + np.cos(m1) * (p.L3 * np.cos(m23) - p.d3 * np.sin(m23))
    y4 = y3 + np.sin(m1) * (p.L3 * np.cos(m23) - p.d3 * np.sin(m23))
    z4 = z3 + p.L3 * np.sin(m23) + p.d3 * np.cos(m23)
    pts.append([x4, y4, z4]); pts.append([x4, y4, z4 - p.L4]) 
    return np.array(pts)

# ==========================================
# 3. 3D 流水线动画渲染 
# ==========================================
def plot_full_pipeline_animation(q_safe, q_hover1, q_target1, q_hover2, q_target2):
    fig = go.Figure()
    R_max = p.L2 + p.L3
    u, v = np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi / 2, 15) 
    fig.add_trace(go.Surface(
        x=R_max * np.outer(np.cos(u), np.sin(v)), y=R_max * np.outer(np.sin(u), np.sin(v)), z=p.L1 + R_max * np.outer(np.ones(np.size(u)), np.cos(v)), 
        opacity=0.1, colorscale='Blues', showscale=False, name='工作空间'
    ))

    key_angles = [q_safe, q_hover1, q_target1, q_hover1, q_safe, q_hover2, q_target2, q_hover2, q_safe]
    tcp_pts = [forward_kinematics_3d(q)[-1] for q in key_angles]
    fig.add_trace(go.Scatter3d(
        x=[pt[0] for pt in tcp_pts], y=[pt[1] for pt in tcp_pts], z=[pt[2] for pt in tcp_pts],
        mode='lines+markers', line=dict(color='red', width=4, dash='dash'), marker=dict(size=6, color='red'), name='预计轨迹'
    ))

    init_pts = forward_kinematics_3d(q_safe)
    fig.add_trace(go.Scatter3d(
        x=init_pts[:,0], y=init_pts[:,1], z=init_pts[:,2],
        mode='lines+markers', line=dict(color='darkblue', width=10),
        marker=dict(size=8, color=['black', 'orange', 'orange', 'orange', 'green']), name='数字孪生实体'
    ))

    frames, frame_idx = [], 0
    def add_segment_frames(q_start, q_end, steps):
        nonlocal frame_idx
        for i in range(steps):
            s = (i / (steps - 1)) ** 2 * (3 - 2 * (i / (steps - 1))) 
            pts = forward_kinematics_3d(q_start + (q_end - q_start) * s)
            frames.append(go.Frame(data=[go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2])], name=str(frame_idx), traces=[2]))
            frame_idx += 1

    add_segment_frames(q_safe, q_hover1, 15); add_segment_frames(q_hover1, q_target1, 8); add_segment_frames(q_target1, q_hover1, 8); add_segment_frames(q_hover1, q_safe, 15)
    add_segment_frames(q_safe, q_hover2, 15); add_segment_frames(q_hover2, q_target2, 8); add_segment_frames(q_target2, q_hover2, 8); add_segment_frames(q_hover2, q_safe, 15)   

    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="▶ 预览 3D 动作序列", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}])], x=0.05, y=0.1)],
        scene=dict(xaxis=dict(range=[-0.3, 0.3], title='X (m)'), yaxis=dict(range=[-0.3, 0.3], title='Y (m)'), zaxis=dict(range=[0, 0.4], title='Z (m)'), aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0), height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# ==========================================
# 4. Streamlit UI 界面与【MQTT 云端通信】
# ==========================================
st.set_page_config(page_title="公网云端 3D 可视化", layout="wide")
st.title("🏭 基于LVGL与云端的模块化机械臂分拣系统 (MQTT版)")

# --- 侧边栏：云端 MQTT 服务器配置 ---
with st.sidebar:
    st.header("☁️ 云端链路配置")
    mqtt_broker = st.text_input("MQTT 服务器地址", value="broker.emqx.io")
    mqtt_port = st.number_input("端口号", value=1883, step=1)
    # ⚠️ 建议把下面这个 Topic 改成包含你名字拼音的独特字符串，防止被别人干扰
    mqtt_topic = st.text_input("发布主题 (Topic)", value="biyeshe_robot_arm_2026/target")
    
    if st.button("🔌 测试云端连通性", width="stretch"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            if s.connect_ex((mqtt_broker, int(mqtt_port))) == 0:  
                st.success(f"✅ 成功连接到云端 {mqtt_broker}")
            else: 
                st.error("❌ 无法连接到服务器，请检查网络")
            s.close()
        except: 
            st.error("❌ 网络异常")

col1, col2 = st.columns([1, 1.8])

with col1:
    st.subheader("🎯 任务坐标参数设定")
    with st.expander("【目标点 1】", expanded=True):
        x1 = st.slider("X1 (m)", -0.250, 0.250, 0.150, step=0.001)
        y1 = st.slider("Y1 (m)", -0.250, 0.250, 0.050, step=0.001)
        z1 = st.slider("Z1 (m)", -0.050, 0.250, 0.050, step=0.001)
    
    with st.expander("【目标点 2】", expanded=True):
        x2 = st.slider("X2 (m)", -0.250, 0.250, 0.150, step=0.001)
        y2 = st.slider("Y2 (m)", -0.250, 0.250, -0.050, step=0.001)
        z2 = st.slider("Z2 (m)", -0.050, 0.250, 0.050, step=0.001)

    hover_offset = 0.05 
    ans_h1, ans_t1 = calculate_4dof_ik(x1, y1, z1 + hover_offset), calculate_4dof_ik(x1, y1, z1)
    ans_h2, ans_t2 = calculate_4dof_ik(x2, y2, z2 + hover_offset), calculate_4dof_ik(x2, y2, z2)
    
    st.divider()
    
    if all(ans[0] is not None for ans in [ans_h1, ans_t1, ans_h2, ans_t2]):
        st.success("✅ 网页端碰撞体积校验通过。")
        
        # 核心：通过 MQTT 将坐标推送到公网
        if st.button("🚀 下发【MQTT坐标】至云端", type="primary", width="stretch"):
            payload = {"t1": {"x": x1, "y": y1, "z": z1}, "t2": {"x": x2, "y": y2, "z": z2}}
            try:
                json_str = json.dumps(payload, separators=(',', ':'))
                
                # 连接并发布
                client = mqtt.Client()
                client.connect(mqtt_broker, mqtt_port, 60)
                client.publish(mqtt_topic, json_str)
                client.disconnect()
                
                st.toast("坐标已成功发送至云端服务器！")
            except Exception as e:
                st.error(f"下发失败：{e}")
                
            st.write(f"📦 已推送到 `{mqtt_topic}` 的数据：")
            st.code(json_str)
    else:
        st.error("⚠️ 物理限界警报: STM32 将无法解算该点位！")

with col2:
    st.subheader("👁️ 3D 机械臂动作预览")
    if all(ans[0] is not None for ans in [ans_h1, ans_t1, ans_h2, ans_t2]):
        q_safe = np.array([0.0, -24.0, -41.0, 75.0])
        fig = plot_full_pipeline_animation(q_safe, ans_h1[0], ans_t1[0], ans_h2[0], ans_t2[0])
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("请调整坐标至工作空间内。")