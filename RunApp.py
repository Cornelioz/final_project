import streamlit as st
import subprocess

def run_streamlit_app():
    subprocess.Popen(["streamlit", "run", "app.py"])

run_streamlit_app()