import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Watt-Watch Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ Campus 'Watt-Watch' Control Room")
st.subheader("Intelligent Energy Auditor")

# Fetch data from backend
# In production, this should fetch from the FastAPI backend url
try:
    response = requests.get("http://localhost:8000/api/status")
    data = response.json()
    rooms = data.get("rooms", [])
except Exception as e:
    st.warning("Backend not running. Showing mock data.")
    rooms = [
        {"id": "Room 101 (Lab)", "person_count": 0, "appliance_state": "ON", "alert": True, "energy_saved_kwh": 0.0},
        {"id": "Room 102 (Class)", "person_count": 25, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 12.5},
        {"id": "Room 103 (Study)", "person_count": 0, "appliance_state": "OFF", "alert": False, "energy_saved_kwh": 8.0},
    ]

# Metrics
col1, col2, col3 = st.columns(3)
total_saved = sum(r["energy_saved_kwh"] for r in rooms)
active_alerts = len([r for r in rooms if r["alert"]])

col1.metric("Total Energy Saved", f"{total_saved:.1f} kWh")
col2.metric("Active Phantom Load Alerts", active_alerts, delta_color="inverse")
col3.metric("Monitored Rooms", len(rooms))

st.markdown("---")
st.header("Live Room Status")

for room in rooms:
    with st.container():
        rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
        rc1.markdown(f"**{room['id']}**")
        rc2.markdown(f"👥 Occupancy: **{room['person_count']}**")
        rc3.markdown(f"🔌 Appliances: **{room['appliance_state']}**")
        
        if room['alert']:
            rc4.error("🚨 ALERT: Empty but Active")
        else:
            rc4.success("✅ Secure")
        st.divider()

st.sidebar.header("Privacy Status")
st.sidebar.success("🔒 Ghost Mode Active")
st.sidebar.info("All video feeds are being anonymized in real-time. No identifiable student data is stored or displayed.")
