# Copyright (c) 2026 Ashwin Ramaseshan. All rights reserved.
# Licensed under the GNU General Public License v3.0
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import tempfile
import cv2        
import math       
from google import genai
from google.genai import types
import numpy as np
import random
import streamlit.components.v1 as components
import io
from moviepy import VideoFileClip 
import concurrent.futures 

import streamlit as st

# Securely pull the key from the hidden local vault (or Cloud vault later)
master_key = st.secrets["master_key"]

API_KEYS = [master_key] if master_key else []

st.set_page_config(page_title="Elite Tactical Scout AI", layout="wide")

valid_keys = [k.strip() for k in API_KEYS if k.strip() and "YOUR_" not in k]

# Initialize Session States
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.raw_data = []
    st.session_state.color_a = "Team A"
    st.session_state.color_b = "Team B"
    st.session_state.team_a = "Team A"
    st.session_state.team_b = "Team B"
    st.session_state.view_mode = 'dashboard' 
    st.session_state.ai_report = None
    st.session_state.history = []

st.title("⚽ AI Tactical Coaching Dashboard")

# ==========================================
# METHODOLOGY TEXT 
# ==========================================
methodology_text = """
### 📊 Tactical Glossary & Metrics

This dashboard utilizes a custom logic engine to transform raw video data into actionable tactical insights. Below is the methodology for each core metric:

**Match Dominance (Net Attacking Threat Score)**
A zero-sum mathematical momentum calculation. It computes a team's real-time threat by assigning distinct weighted values to their pitch position (Attacking Third = Highest), attacking tempo (Fast Direct = Highest), and spatial half-space occupancy. A heavy +5 point multiplier is applied strictly to high-value counter-attacks and fast vertical transitions. Only the team controlling the primary possession registers a threat score for that minute.

**Average Threat Score**
Calculates the mean value of the Threat Score exclusively during the minutes a team held possession. 

**Primary Zone**
Identifies the specific third of the pitch (Defensive, Middle, or Attacking) where a team logged the highest volume of sustained possession over the course of the match segment.

**Time in Attack**
Evaluates the literal fractional minutes a team spent actively establishing possession and attacking inside the opponent's defensive third.

**Active Possession Distribution (Heatmap)**
Measures the territorial footprints of both teams across all three zones. *Note: This does not necessarily sum to 90 minutes.* It explicitly filters out purely neutral moments, rapid transitions, or major stoppages to provide a pure reflection of established, active possession.

**Defensive Block Height (Positioning)**
Measures the *starting position* of a team's defensive wall out of possession:
* **Low Block:** Retreating deep into their own penalty area to defend.
* **Mid Block:** Holding their defensive line near the midfield circle.
* **High Block:** Pushing defenders aggressively high up the pitch into the opponent's half.

**Pressing Intensity (1-10 Scale)**
An AI-calculated metric measuring the aggression of the team's tackling and closing down. (1-3: Passive, 4-6: Engaging near halfway line, 7-10: Aggressive).

**Attacking Bias (Geometry)**
Calculates the exact sequence frequency a team successfully funneled an attack down a specific geometric channel. Sterile defensive possession is filtered out. The 'Central Channel' is strictly geometrically defined as the exact physical width of the 18-yard penalty box.

---
### 🧠 The Backend Architecture & Prompt-Based Fine-Tuning
This dashboard is powered by a custom multimodal data pipeline that converts raw match footage into structured, mathematical tactical arrays using Google's **Gemini 2.5 Flash** vision-language model.

**1. Local Micro-Batching Protocol**
The Python backend physically slices the match into strict 1-minute intervals locally, uploading and processing them sequentially to guarantee the AI maintains "tunnel vision" on specific tactical actions.

**2. Architectural Constraint Fine-Tuning**
The model is strictly fine-tuned through systemic prompt architecture and logic locks to avoid common vision-model hallucinations (e.g., center-frame bias).

---
### 👨‍💻 About the Creator
I am a Master’s student in **Applied Statistics and Data Science at UCLA** and a UC Riverside Alumni in Computer Science. My technical expertise sits at the intersection of AI, Machine Learning, and Full-Stack Data Science. 
🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/ashwin-ramaseshan-a63188201/)
"""

def get_pdf_download_button():
    pdf_path = "Tactical_Vision.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        return st.download_button(
            label="📄 Download Full Project Report (PDF)",
            data=pdf_bytes,
            file_name="Tactical_Vision.pdf",
            mime="application/pdf",
        )
    else:
        return st.warning(f"Project Report PDF not found at {pdf_path}. Please check the file path.")

def process_single_minute(start_min, duration_sec, temp_video_path, api_key, master_team_a_color, master_team_b_color):
    client = genai.Client(api_key=api_key)
    end_min = min(start_min + 1, duration_sec / 60.0)
    
    chunk_path = f"temp_chunk_{start_min}.mp4"
    try:
        with VideoFileClip(temp_video_path) as clip:
            subclip = clip.subclipped(start_min * 60, end_min * 60)
            subclip.write_videofile(chunk_path, codec="libx264", audio=False, logger=None)
    except Exception as e:
        return {"error": f"Slicing failed: {e}", "minute": start_min}

    try:
        video_file = client.files.upload(file=chunk_path)
        timeout_counter = 0
        max_checks = 30 
        while video_file.state.name == "PROCESSING":
            if timeout_counter >= max_checks:
                try: client.files.delete(name=video_file.name)
                except: pass
                return {"error": "Google Cloud processing timed out.", "minute": start_min}
            time.sleep(3)
            video_file = client.files.get(name=video_file.name)
            timeout_counter += 1
            
        if video_file.state.name == "FAILED":
            return {"error": "Cloud processing failed.", "minute": start_min}
    except Exception as e:
         return {"error": f"Upload failed: {e}", "minute": start_min}

    if master_team_a_color is None:
        color_instruction = "1. Identify the two teams by their primary kit colors. Label them Team A and Team B."
    else:
        color_instruction = f"1. CRITICAL: Team A kit color is exactly '{master_team_a_color}' and Team B kit color is exactly '{master_team_b_color}'."

    data_prompt = f"""
    Watch this 1-minute soccer match clip.
    {color_instruction}
    2. Output EXACTLY 1 JSON object describing the tactics for this specific 60-second window. Do not output an array.
    
    POSSESSION RULE (MAJORITY RULES): Assign 'team_in_possession' to the team that held the ball for the absolute majority of this 60-second interval. Do NOT award dual possession. 
    
    HIGH-VALUE STEAL EXCEPTION: If the defending team steals the ball and registers a clear goal-scoring threat or shot inside the opponent's penalty box during this minute, they "steal" the minute. You MUST assign 'team_in_possession' to the counter-attacking team and assign them a 'fast_direct' tempo.
    
    ZONAL MAJORITY RULE: 'ball_zone' MUST strictly reflect where the ball spent the majority of the 60 seconds relative to the team in possession. If a team defends deep for 55 seconds and counters for 5 seconds, the zone is 'defensive_third'.
    
    FRACTIONAL ATTACKING TIME: You must output two integers (0-60) for 'team_a_attack_sec' and 'team_b_attack_sec' representing exactly how many literal seconds each team spent in the attacking third during this minute.
    
    CRITICAL VOCABULARY UPGRADE (GEGENPRESSING): If a team immediately swarms the ball high up the pitch after losing it, you MUST tag their 'pressing_trigger' as 'gegenpress' and their 'block_height' as 'high'. Do this EVEN IF the opponent bypasses the press and forces them to defend deep later in the minute. Do not fall for the "Recovery Illusion."
    
    CRITICAL VOCABULARY UPGRADE (TRANSITIONS): A 'counter_attack' is when a team sits deep, wins the ball, and breaks. A 'fast_vertical_transition' is when a team uses rapid, direct passing to bypass an opponent's high press. Use these tags correctly in 'transition_threat'.
    
    CRITICAL VOCABULARY UPGRADE (POSSESSION): If a team's primary goal is to pin the opponent in their own half (Zonal Possession), you MUST tag their 'attacking_tempo' as 'sustained_high_pressure', not just generic 'patient_possession'.
    
    PHASE 2 FIX - GEOMETRIC DEFINITION OF CENTRAL CHANNEL: The 'central_channel' is STRICTLY defined as the physical width of the 18-yard penalty box. If the ball is operating outside the width of the 18-yard box, you MUST classify the 'attacking_bias' as 'left_flank' or 'right_flank'. Do not fall for the broadcast camera illusion.
    
    EFFECTIVE PLAYING TIME INSTRUCTION: If the majority of the minute is spent dealing with an injury, a player walking to set up a corner, or extreme time-wasting, set BOTH teams' attacking tempo to "dead_ball_stoppage". 

    Global Variables:
    - "team_a_color": string
    - "team_b_color": string
    - "team_in_possession": color of the ONE team with primary ball control (Apply exception rule if needed)
    - "ball_zone": "defensive_third", "middle_third", or "attacking_third" (Where ball spent the majority of the minute)
    - "team_a_attack_sec": Integer (0-60)
    - "team_b_attack_sec": Integer (0-60)
    
    Team A Variables:
    - "team_a_pressing_intensity": Integer from 1 to 10
    - "team_a_block_height": "low", "mid", or "high"
    - "team_a_half_space_occupancy": Integer (0 to 4 players)
    - "team_a_vertical_compactness": "tight", "standard", or "stretched"
    - "team_a_build_up_shape": "3-2", "2-3", "4-2", or "3-box-3"
    - "team_a_attacking_tempo": "fast_direct", "patient_possession", "sustained_high_pressure", "none", or "dead_ball_stoppage"
    - "team_a_transition_threat": "counter_attack", "fast_vertical_transition", "sustained_build", or "none"
    - "team_a_striker_profile": "false_9", "target_man", or "channel_runner"
    - "team_a_fullback_role": "overlapping", "inverted", or "defensive"
    - "team_a_pressing_trigger": "gegenpress", "loss_of_possession", "backward_pass", "poor_touch", or "none"
    - "team_a_rest_defense_shape": "3-2", "2-3", or "unstructured"
    - "team_a_attacking_bias": "left_flank", "right_flank", or "central_channel"
    - "team_a_defensive_line_action": "drop_deep" or "step_up"
    
    Team B Variables:
    - "team_b_pressing_intensity": Integer from 1 to 10
    - "team_b_block_height": "low", "mid", or "high"
    - "team_b_half_space_occupancy": Integer (0 to 4 players)
    - "team_b_vertical_compactness": "tight", "standard", or "stretched"
    - "team_b_build_up_shape": "3-2", "2-3", "4-2", or "3-box-3"
    - "team_b_attacking_tempo": "fast_direct", "patient_possession", "sustained_high_pressure", "none", or "dead_ball_stoppage"
    - "team_b_transition_threat": "counter_attack", "fast_vertical_transition", "sustained_build", or "none"
    - "team_b_striker_profile": "false_9", "target_man", or "channel_runner"
    - "team_b_fullback_role": "overlapping", "inverted", or "defensive"
    - "team_b_pressing_trigger": "gegenpress", "loss_of_possession", "backward_pass", "poor_touch", or "none"
    - "team_b_rest_defense_shape": "3-2", "2-3", or "unstructured"
    - "team_b_attacking_bias": "left_flank", "right_flank", or "central_channel"
    - "team_b_defensive_line_action": "drop_deep" or "step_up"
    """
    
    max_retries = 3 
    minute_data = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[video_file, data_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            minute_data = json.loads(response.text)
            if isinstance(minute_data, list) and len(minute_data) > 0:
                minute_data = minute_data[0]
                
            minute_data["timestamp"] = f"{start_min:02d}:00-{(start_min+1):02d}:00"
            minute_data["_minute_index"] = start_min 
            break 
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3) 
                continue
            else:
                minute_data = {"error": "Parsing failed", "minute": start_min}
    
    try: client.files.delete(name=video_file.name)
    except: pass
    if os.path.exists(chunk_path): os.remove(chunk_path)
        
    return minute_data

# ==========================================
# STEP 1: UI RENDER
# ==========================================
if st.session_state.step == 1:
    
    sidebar_ui = st.sidebar.empty()
    
    with sidebar_ui.container():
        st.header("1. Upload Match Footage (Live Engine)")
        uploaded_video = st.file_uploader("Choose a video file (Max 2GB)", type=["mp4", "mov", "avi"])
        st.caption("*45 mins max. Requires active Gemini API key.*")
        extract_btn = st.button("Extract Tactical Data")
        
        # --- NEW: RECRUITER DEMO MODE UI ---
        st.markdown("---")
        st.header("⚡ Recruiter Demo Mode")
        st.write("Upload a pre-processed `demo_data.csv` file to instantly bypass the 15-minute AI video processing phase.")
        uploaded_csv = st.file_uploader("Upload Tactical CSV", type=["csv"])
        demo_btn = st.button("Load Instant Demo Match")
        
        if len(st.session_state.history) > 0:
            st.markdown("---")
            st.subheader("📁 Past Reports")
            history_options = {"(Select a past report...)": -1}
            for i, h in enumerate(st.session_state.history):
                history_options[f"Match {i+1}: {h['team_a']} vs {h['team_b']}"] = i
                
            selected_history = st.selectbox("Load History", list(history_options.keys()))
            if history_options[selected_history] != -1:
                h_idx = history_options[selected_history]
                st.session_state.team_a = st.session_state.history[h_idx]['team_a']
                st.session_state.team_b = st.session_state.history[h_idx]['team_b']
                st.session_state.color_a = st.session_state.history[h_idx]['color_a']
                st.session_state.color_b = st.session_state.history[h_idx]['color_b']
                st.session_state.raw_data = st.session_state.history[h_idx]['raw_data']
                st.session_state.ai_report = st.session_state.history[h_idx]['ai_report']
                st.session_state.step = 3
                st.session_state.view_mode = 'dashboard'
                st.rerun()

    # --- DEMO MODE EXECUTION ---
    if demo_btn:
        if uploaded_csv is None:
            st.error("⚠️ Please upload a CSV file first to use Demo Mode.")
        else:
            try:
                demo_df = pd.read_csv(uploaded_csv)
                st.session_state.raw_data = demo_df.to_dict('records')
                
                # Try to grab colors if they exist in the CSV
                if 'team_a_color' in demo_df.columns:
                    st.session_state.color_a = str(demo_df['team_a_color'].iloc[0]).title()
                    st.session_state.color_b = str(demo_df['team_b_color'].iloc[0]).title()
                else:
                    st.session_state.color_a = "Team A"
                    st.session_state.color_b = "Team B"
                    
                st.session_state.step = 2
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error reading CSV file: {e}")

    # --- LIVE EXTRACTION EXECUTION ---
    if extract_btn:
        if not valid_keys:
            st.error("Please add your Gemini API Key in the configuration section.")
        elif uploaded_video is None:
            st.error("Please upload a video file.")
        else:
            sidebar_ui.empty()
            
            progress_bar = st.progress(0, text="🚀 Our AI Scout is currently watching the match... This will take a few minutes.")
            game_placeholder = st.empty()
            status_text = st.empty()
            
            carousel_html = """
            <div style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center;">
                <div style="width: 100%; position: relative; border-radius: 15px; border: 3px solid #b2dfdb; overflow: hidden; box-shadow: inset 0px 0px 15px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    
                    <button onclick="changeGame(-1)" style="position: absolute; left: 15px; top: 50%; transform: translateY(-50%); z-index: 100; background: #333; color: white; border: none; font-size: 24px; cursor: pointer; border-radius: 50%; width: 45px; height: 45px; display: flex; justify-content: center; align-items: center; opacity: 0.8;">&#10094;</button>
                    <button onclick="changeGame(1)" style="position: absolute; right: 15px; top: 50%; transform: translateY(-50%); z-index: 100; background: #333; color: white; border: none; font-size: 24px; cursor: pointer; border-radius: 50%; width: 45px; height: 45px; display: flex; justify-content: center; align-items: center; opacity: 0.8;">&#10095;</button>

                    <div id="game1" class="game-view" style="display: flex; width: 100%; height: 350px; background: linear-gradient(to bottom, #87CEEB, #e0f6ff); position: relative; flex-direction: column; align-items: center;">
                        <h2 style="color: #004d40; margin-top: 30px; margin-bottom: 5px; user-select: none;">Keep the ball up</h2>
                        <p style="color: #555; font-size: 14px; margin: 0; user-select: none;">How to Play: Click the ball or press Spacebar to keep it in the air!</p>
                        <p id="score1" style="font-size: 28px; font-weight: bold; color: #00695c; margin-top: 10px; user-select: none;">Score: 0</p>
                        <div id="ball1" style="font-size: 45px; position: absolute; top: 150px; left: 50%; transform: translateX(-50%); cursor: pointer; user-select: none; transition: transform 0.1s;">⚽</div>
                    </div>

                    <div id="game2" class="game-view" style="display: none; width: 100%; height: 350px; background: linear-gradient(to bottom, #2e7d32, #4db6ac); position: relative; flex-direction: column; align-items: center; overflow: hidden;">
                        <h2 style="color: #ffffff; margin-top: 30px; margin-bottom: 5px; user-select: none; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);"> Goalkeeper</h2>
                        <p style="color: #e0e0e0; font-size: 14px; margin: 0; user-select: none;">How to Play: Use Left/Right arrow keys to save the shots!</p>
                        <p id="score2" style="font-size: 28px; font-weight: bold; color: #ffffff; margin-top: 10px; user-select: none; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Saves: 0</p>
                        <div style="width: 80%; height: 120px; border: 5px solid white; border-bottom: none; position: absolute; bottom: 40px; left: 10%;"></div>
                        
                        <div id="ball2" style="font-size: 35px; position: absolute; top: -50px; left: 50%; transform: translateX(-50%); user-select: none;">⚽</div>
                        <div id="goalie" style="font-size: 50px; position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); user-select: none; transition: left 0.1s linear;">🧤</div>
                    </div>

                </div>

                <div id="quoteDisplay" style="background: rgba(245, 245, 245, 0.9); padding: 15px 25px; border-radius: 10px; text-align: center; font-style: italic; color: #444; font-size: 16px; border-left: 5px solid #d9383a; transition: opacity 0.5s; width: 80%;">
                    Loading...
                </div>
            </div>

            <script>
                const quotes = [
                    '"I am not a perfectionist, but I like to feel that things are done well." – Cristiano Ronaldo',
                    '"Playing football is very simple, but playing simple football is the hardest thing there is." – Johan Cruyff',
                    '"The objective is to move the opponent, not the ball." – Pep Guardiola',
                    '"Without the ball, you can\\'t win. With the ball, you have a chance." – Xavi Hernandez',
                    '"If you control the midfield, you control the game." – Sir Alex Ferguson'
                ];
                let quoteIndex = 0;
                const quoteDisplay = document.getElementById('quoteDisplay');
                
                function updateQuote() {
                    quoteDisplay.style.opacity = 0;
                    setTimeout(() => {
                        quoteDisplay.innerHTML = "<b>" + quotes[quoteIndex] + "</b>";
                        quoteDisplay.style.opacity = 1;
                        quoteIndex = (quoteIndex + 1) % quotes.length;
                    }, 500);
                }
                updateQuote();
                setInterval(updateQuote, 10000); 

                let currentGame = 1;
                function changeGame(dir) {
                    document.getElementById('game' + currentGame).style.display = 'none';
                    currentGame += dir;
                    if(currentGame > 2) currentGame = 1;
                    if(currentGame < 1) currentGame = 2;
                    document.getElementById('game' + currentGame).style.display = 'flex';
                }

                const ball1 = document.getElementById('ball1');
                const score1 = document.getElementById('score1');
                let y1 = 150, v1 = 0, g1 = 0.35, s1 = 0, anim1 = true;
                
                function kick1() { v1 = -8; s1++; score1.innerText = "Score: " + s1; }
                ball1.addEventListener('mousedown', kick1);
                
                function loop1() {
                    if(!anim1) return;
                    v1 += g1; y1 += v1;
                    if(y1 > 280) { y1 = 280; if(s1>0) {score1.innerText = "Dropped! Score: "+s1; s1=0; v1=0;} }
                    if(y1 < 0) { y1 = 0; v1 *= -0.5; }
                    ball1.style.top = y1 + 'px';
                    requestAnimationFrame(loop1);
                }
                loop1();

                const goalie = document.getElementById('goalie');
                const ball2 = document.getElementById('ball2');
                const score2 = document.getElementById('score2');
                let gX = 50; 
                let b2X = 50, b2Y = -50; 
                let b2Vy = 3.5;
                let s2 = 0;
                
                function resetBall2() {
                    b2Y = -50;
                    b2X = 20 + Math.random() * 60; 
                    b2Vy = 3.5 + (s2 * 0.3); 
                }
                
                function loop2() {
                    b2Y += b2Vy;
                    
                    if (b2Y > 260) { 
                        if (Math.abs(b2X - gX) < 12) { 
                            s2++;
                            score2.innerText = "Saves: " + s2;
                            resetBall2();
                        } else {
                            if(s2 > 0) score2.innerText = "Goal! Final Saves: " + s2;
                            s2 = 0;
                            resetBall2();
                        }
                    }
                    ball2.style.top = b2Y + 'px';
                    ball2.style.left = b2X + '%';
                    requestAnimationFrame(loop2);
                }
                resetBall2();
                loop2();

                document.addEventListener('keydown', function(event) {
                    if (event.code === 'Space' && currentGame === 1) {
                        event.preventDefault(); 
                        kick1();
                    }
                    
                    if (currentGame === 2) {
                        if (event.code === 'ArrowLeft') { 
                            event.preventDefault();
                            gX -= 7; 
                            if(gX < 15) gX = 15; 
                            goalie.style.left = gX + '%'; 
                        }
                        if (event.code === 'ArrowRight') { 
                            event.preventDefault();
                            gX += 7; 
                            if(gX > 85) gX = 85; 
                            goalie.style.left = gX + '%'; 
                        }
                    }
                });
            </script>
            """
            with game_placeholder:
                components.html(carousel_html, height=450)

            status_text.info("Saving master video to temporary storage...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_video_path = tmp_file.name

            status_text.info("Scanning video duration...")
            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0:
                duration_sec = frames / fps
                TOTAL_MINUTES = math.ceil(duration_sec / 60) 
            else:
                TOTAL_MINUTES = 1 
            cap.release()
            
            st.toast(f"Detected video length: {TOTAL_MINUTES} minutes.")

            try:
                all_tactical_data = []
                master_team_a_color = None
                master_team_b_color = None
                
                status_text.info("Establishing Team Colors from Minute 1...")
                first_min_data = process_single_minute(0, duration_sec, temp_video_path, valid_keys[0], None, None)
                
                if "error" not in first_min_data:
                    master_team_a_color = first_min_data.get("team_a_color", "Team A")
                    master_team_b_color = first_min_data.get("team_b_color", "Team B")
                    st.session_state.color_a = str(master_team_a_color).title()
                    st.session_state.color_b = str(master_team_b_color).title()
                    all_tactical_data.append(first_min_data)
                
                if TOTAL_MINUTES > 1:
                    max_workers = 3 
                    completed_count = 1 
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(process_single_minute, start_min, duration_sec, temp_video_path, valid_keys[0], master_team_a_color, master_team_b_color) for start_min in range(1, TOTAL_MINUTES)]
                        
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if "error" not in result:
                                all_tactical_data.append(result)
                                
                            completed_count += 1
                            progress_percentage = int((completed_count / TOTAL_MINUTES) * 100)
                            progress_bar.progress(completed_count / TOTAL_MINUTES, text=f"🚀 Our AI Scout is currently watching the match... ({progress_percentage}% Complete)")
                
                all_tactical_data.sort(key=lambda x: x.get('_minute_index', 0))
                
                for d in all_tactical_data:
                    if '_minute_index' in d:
                        del d['_minute_index']

                progress_bar.progress(1.0, text="Data Extraction: 100% Complete")

                st.session_state.raw_data = all_tactical_data
                
                game_placeholder.empty()
                status_text.success("🚀 Parallel Data Extraction Complete!")
                progress_bar.empty()
                
                st.session_state.step = 2
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass

# ==========================================
# STEP 2: TEAM MAPPING (User Input)
# ==========================================
elif st.session_state.step == 2:
    st.sidebar.header("2. Team Mapping")
    
    st.info("🎯 **TEAM MAPPING**")
    st.write(f"**AI detected Team A as {st.session_state.color_a}.**")
    user_a = st.text_input(f"Enter real team name (or press Enter to keep '{st.session_state.color_a}'):", placeholder=st.session_state.color_a)
    
    st.write(f"**AI detected Team B as {st.session_state.color_b}.**")
    user_b = st.text_input(f"Enter real team name (or press Enter to keep '{st.session_state.color_b}'):", placeholder=st.session_state.color_b)
    
    if st.button("Generate Dashboard"):
        st.session_state.team_a = user_a.strip() if user_a.strip() else st.session_state.color_a
        st.session_state.team_b = user_b.strip() if user_b.strip() else st.session_state.color_b
        st.session_state.step = 3
        st.session_state.ai_report = None 
        st.session_state.view_mode = 'dashboard'
        st.rerun()

# ==========================================
# STEP 3: MATH, AI DIAGNOSIS & DASHBOARD
# ==========================================
elif st.session_state.step == 3:
    
    df = pd.DataFrame(st.session_state.raw_data)
    
    # Catch bad/missing API responses before they crash Pandas
    if 'team_in_possession' not in df.columns:
        st.error("⚠️ AI Extraction Failed: The cloud API timed out or returned malformed data. If you are a recruiter reviewing this portfolio, please use the 'Quick Demo Mode' on the home screen.")
        if st.button("⬅️ Start Over"):
            st.session_state.step = 1
            st.session_state.raw_data = []
            st.rerun()
        st.stop()

    team_a = st.session_state.team_a
    team_b = st.session_state.team_b
    color_a = st.session_state.color_a
    color_b = st.session_state.color_b
    ai_report_text = st.session_state.ai_report

    if len(st.session_state.history) > 0 and 'load_idx' in st.session_state:
         pass 

    df.replace(["", " ", "Unknown", "N/A", None], np.nan, inplace=True)
    df = df.ffill().bfill() 

    df['team_a_has_ball'] = (df['team_in_possession'].str.lower() == color_a.lower()).astype(int)
    df['team_b_has_ball'] = (df['team_in_possession'].str.lower() == color_b.lower()).astype(int)

    df['team_a_trans_threat'] = df.get('team_a_transition_threat', pd.Series(['none']*len(df))).str.lower().str.strip()
    df['team_b_trans_threat'] = df.get('team_b_transition_threat', pd.Series(['none']*len(df))).str.lower().str.strip()

    ta_counters = int(df['team_a_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']).sum())
    tb_counters = int(df['team_b_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']).sum())

    zone_map = {'attacking_third': 3, 'middle_third': 1.5, 'defensive_third': 0.5}
    df['zone_numeric'] = df.get('ball_zone', pd.Series(['middle_third']*len(df))).map(zone_map).fillna(1.5)

    tempo_map = {'fast_direct': 2, 'sustained_high_pressure': 2, 'patient_possession': 1, 'none': 0, 'static': 0, 'dead_ball_stoppage': 0}
    df['team_a_tempo_num'] = df.get('team_a_attacking_tempo', pd.Series(['none']*len(df))).map(tempo_map).fillna(1)
    df['team_b_tempo_num'] = df.get('team_b_attacking_tempo', pd.Series(['none']*len(df))).map(tempo_map).fillna(1)

    team_a_hs = pd.to_numeric(df.get('team_a_half_space_occupancy', 0), errors='coerce').fillna(0)
    team_b_hs = pd.to_numeric(df.get('team_b_half_space_occupancy', 0), errors='coerce').fillna(0)

    ta_counter_boost = np.where(df['team_a_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)
    tb_counter_boost = np.where(df['team_b_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)

    df['team_a_raw_threat'] = df['team_a_has_ball'] * ((df['zone_numeric'] * 2.0) + (df['team_a_tempo_num'] * 1.5) + (team_a_hs * 1.0) + ta_counter_boost)
    df['team_b_raw_threat'] = df['team_b_has_ball'] * ((df['zone_numeric'] * 2.0) + (df['team_b_tempo_num'] * 1.5) + (team_b_hs * 1.0) + tb_counter_boost)

    df['net_momentum'] = df['team_a_raw_threat'] - df['team_b_raw_threat']
    df['smoothed_net_momentum'] = df['net_momentum'].rolling(window=2, min_periods=1).mean().round(2)

    team_a_avg_dom = round(df[df['smoothed_net_momentum'] > 0]['smoothed_net_momentum'].mean(), 2) if not df[df['smoothed_net_momentum'] > 0].empty else 0
    team_b_avg_dom = round(abs(df[df['smoothed_net_momentum'] < 0]['smoothed_net_momentum'].mean()), 2) if not df[df['smoothed_net_momentum'] < 0].empty else 0
    
    team_a_peak_idx = df['smoothed_net_momentum'].idxmax()
    team_a_peak_time = df.loc[team_a_peak_idx, 'timestamp'] if pd.notna(team_a_peak_idx) else "Unknown"
    
    team_b_peak_idx = df['smoothed_net_momentum'].idxmin()
    team_b_peak_time = df.loc[team_b_peak_idx, 'timestamp'] if pd.notna(team_b_peak_idx) else "Unknown"

    ta_att_sec = pd.to_numeric(df.get('team_a_attack_sec', 0), errors='coerce').fillna(0).sum()
    tb_att_sec = pd.to_numeric(df.get('team_b_attack_sec', 0), errors='coerce').fillna(0).sum()
    
    ta_att_count = round(ta_att_sec / 60, 1)
    tb_att_count = round(tb_att_sec / 60, 1)

    def get_mode(column): return df[column].mode()[0] if column in df.columns and not df[column].dropna().empty else "Unknown"
    
    ta_block_height = get_mode('team_a_block_height')
    ta_press_intensity = get_mode('team_a_pressing_intensity')
    ta_build_up = get_mode('team_a_build_up_shape')
    ta_tempo = get_mode('team_a_attacking_tempo')
    ta_trigger = get_mode('team_a_pressing_trigger')
    ta_striker = get_mode('team_a_striker_profile')
    ta_fullback = get_mode('team_a_fullback_role')
    ta_bias = get_mode('team_a_attacking_bias')
    ta_def_action = get_mode('team_a_defensive_line_action')
    ta_half_space = get_mode('team_a_half_space_occupancy')
    
    tb_block_height = get_mode('team_b_block_height')
    tb_press_intensity = get_mode('team_b_pressing_intensity')
    tb_build_up = get_mode('team_b_build_up_shape')
    tb_tempo = get_mode('team_b_attacking_tempo')
    tb_trigger = get_mode('team_b_pressing_trigger')
    tb_striker = get_mode('team_b_striker_profile')
    tb_fullback = get_mode('team_b_fullback_role')
    tb_bias = get_mode('team_b_attacking_bias')
    tb_def_action = get_mode('team_b_defensive_line_action')
    tb_half_space = get_mode('team_b_half_space_occupancy')

    sidebar_placeholder = st.sidebar.empty()

    if st.session_state.ai_report is None and st.session_state.view_mode != 'ticker':
        dashboard_spinner = st.empty()
        with dashboard_spinner.container():
            with st.spinner("Running universal tactical engine and generating deep AI report..."):
                client = genai.Client(api_key=valid_keys[0]) 
                
                writing_prompt = f"""
                Act as an elite soccer tactical data scientist. You have been handed a statistical readout from a computer vision model that analyzed a segment of a match between {team_a} and {team_b}. 
                
                YOUR MISSION: 
                1. DIAGNOSE THE TACTICS: Look at the raw metrics below and use your vast knowledge of soccer strategy to figure out what tactical systems both teams were employing.
                2. WRITE THE REPORT: Explain to the coaching staff exactly HOW they are executing these systems mechanically.
                
                THE HARD DATA:
                - {team_a} Avg Threat Score: {team_a_avg_dom}/10 (Peak at: {team_a_peak_time})
                - {team_b} Avg Threat Score: {team_b_avg_dom}/10 (Peak at: {team_b_peak_time})
                - Total Fractional Time in Attack (Minutes): {team_a} ({ta_att_count}), {team_b} ({tb_att_count})
                - Total High-Value Transitions (Counters/Fast Vertical): {team_a} ({ta_counters}), {team_b} ({tb_counters})

                PREDOMINANT TACTICAL MODES:
                - {team_a}: Def: Block: {ta_block_height} | Press: {ta_press_intensity}/10 | Line: {ta_def_action} | Trigger: {ta_trigger}. Off: Build: {ta_build_up} | Tempo: {ta_tempo} | Bias: {ta_bias} | Striker: {ta_striker}. Space: Half-Space: {ta_half_space} | Fullbacks: {ta_fullback}
                - {team_b}: Def: Block: {tb_block_height} | Press: {tb_press_intensity}/10 | Line: {tb_def_action} | Trigger: {tb_trigger}. Off: Build: {tb_build_up} | Tempo: {tb_tempo} | Bias: {tb_bias} | Striker: {tb_striker}. Space: Half-Space: {tb_half_space} | Fullbacks: {tb_fullback}

                CRITICAL INSTRUCTION FOR WRITING: 
                - For EVERY single section and sub-section below, you MUST write a rich, highly detailed analytical paragraph (at least 4-5 sentences). 
                - DO NOT use brief bullet points. Expand deeply on the tactical theory and what it means for the game.
                - DO NOT use raw variable names or key-value pairs (like 'line: step_up') anywhere in your text. Translate all data into natural, free-flowing, professional scouting language.
                - YOU MUST PLACE A DOUBLE LINE BREAK BETWEEN EVERY SINGLE NUMBERED POINT so the Markdown formats cleanly.
                - TRANSITIONAL THREAT: Analyze the team's transitional threat based on the Total Transitions Logged data. Do NOT explicitly list the raw count of transitions. Instead, write a narrative analysis explaining how they successfully absorbed pressure and used fast vertical transitions or counters to bypass the opponent's structure.

                Format your analysis EXACTLY like this structure:

                ### Tactical Diagnosis & Match Flow
                [Write a massive, comprehensive paragraph identifying the overarching tactical battle and flow of the game. Address if it was one-sided or highly open-ended.]

                ### {team_a} Tactical Profile

                **1. SYSTEM IDENTIFICATION:** [Write a detailed 4-5 sentence paragraph identifying philosophy based on data.]

                **2. POSSESSION & TERRITORY:** [Write a detailed 4-5 sentence paragraph analyzing their Fractional Time in Attack and Build-up.]

                **3. DEFENSIVE MECHANICS:** [Write a detailed 4-5 sentence paragraph analyzing Line Action, Pressing Intensity, Trigger.]

                **4. TRANSITION & ATTACK:** [Write a detailed 4-5 sentence paragraph analyzing Bias, Striker Profile, Tempo, and seamlessly integrate their transitional/counter-attacking strategy without stating the raw number.]

                **5. SPATIAL MANIPULATION:** [Write a detailed 4-5 sentence paragraph analyzing Half-Space and Fullback Role.]

                **6. VULNERABILITIES:** [Write a detailed 4-5 sentence paragraph explaining the inherent flaws of this system.]

                ### {team_b} Tactical Profile

                **1. SYSTEM IDENTIFICATION:** [Write a detailed 4-5 sentence paragraph identifying philosophy based on data.]

                **2. POSSESSION & TERRITORY:** [Write a detailed 4-5 sentence paragraph analyzing their Fractional Time in Attack and Build-up.]

                **3. DEFENSIVE MECHANICS:** [Write a detailed 4-5 sentence paragraph analyzing Line Action, Pressing Intensity, Trigger.]

                **4. TRANSITION & ATTACK:** [Write a detailed 4-5 sentence paragraph analyzing Bias, Striker Profile, Tempo, and seamlessly integrate their transitional/counter-attacking strategy without stating the raw number.]

                **5. SPATIAL MANIPULATION:** [Write a detailed 4-5 sentence paragraph analyzing Half-Space and Fullback Role.]

                **6. VULNERABILITIES:** [Write a detailed 4-5 sentence paragraph explaining the inherent flaws of this system.]

                ### DATA-DRIVEN COACHING ADJUSTMENTS

                **For {team_b}:** 1. [First specific tactical change paragraph.]

                2. [Second specific tactical change paragraph.]

                **For {team_a}:** 1. [First specific tactical change paragraph.]

                2. [Second specific tactical change paragraph.]
                """
                
                max_report_retries = 5
                for attempt in range(max_report_retries):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=writing_prompt,
                            config=types.GenerateContentConfig(temperature=0.1) 
                        )
                        st.session_state.ai_report = response.text
                        ai_report_text = st.session_state.ai_report
                        break
                    except Exception as e:
                        if attempt < max_report_retries - 1:
                            time.sleep(5)
                            continue
                        else:
                            st.error("❌ Failed to generate report after max retries due to strict API limits.")
                            st.stop()
        dashboard_spinner.empty()
    
    with sidebar_placeholder.container():
        st.success("Analysis Complete!")
        if st.button("💾 Save Report to History"):
            if st.session_state.ai_report is not None:
                st.session_state.history.append({
                    'team_a': st.session_state.team_a,
                    'team_b': st.session_state.team_b,
                    'raw_data': st.session_state.raw_data,
                    'ai_report': st.session_state.ai_report,
                    'color_a': st.session_state.color_a,
                    'color_b': st.session_state.color_b
                })
                st.success("Report Saved!")
                
        if st.button("🗑️ Start Over (Clear Screen)"):
            st.session_state.step = 1
            st.session_state.view_mode = 'dashboard'
            st.session_state.ai_report = None
            st.session_state.raw_data = []
            st.rerun()

        st.markdown("---")
        st.subheader("Navigation")
        nav_selection = st.radio("Go to:", ["Match Dashboard", "Methodology & Project Report"])
        
        if len(st.session_state.history) > 0:
            st.markdown("---")
            st.subheader("📁 Past Reports")
            history_options = {"(Active Session)": -1}
            for i, h in enumerate(st.session_state.history):
                history_options[f"Match {i+1}: {h['team_a']} vs {h['team_b']}"] = i
                
            selected_history = st.selectbox("Load History", list(history_options.keys()))
            if history_options[selected_history] != -1:
                h_idx = history_options[selected_history]
                st.session_state.raw_data = st.session_state.history[h_idx]['raw_data']
                st.session_state.team_a = st.session_state.history[h_idx]['team_a']
                st.session_state.team_b = st.session_state.history[h_idx]['team_b']
                st.session_state.color_a = st.session_state.history[h_idx]['color_a']
                st.session_state.color_b = st.session_state.history[h_idx]['color_b']
                st.session_state.ai_report = st.session_state.history[h_idx]['ai_report']
                st.session_state.load_idx = h_idx
                st.rerun()

    if nav_selection == "Methodology & Project Report":
        get_pdf_download_button()
        st.markdown(methodology_text)

    elif nav_selection == "Match Dashboard":
        
        zone_help = "Shows the pitch zone where the team held the ball most frequently. Attacking Third indicates high pressure, Middle Third reflects controlled build-up, and Defensive Third indicates being pinned back."
        press_help = "Calculated from 1-10 based on defensive line height and intensity. \n\n• 1-3: Low Block (Passive/Defending the box)\n• 4-6: Mid Block (Engaging near halfway line)\n• 7-10: High Press (Aggressive/Hunting the ball)"
        att_time_help = "The true fractional number of minutes this team held sustained possession inside the opponent's defensive third."
        bias_help = "Shows the percentage of successful attacking sequences funneled down the left flank, right flank, or center. Minutes where the team was purely defending have been filtered out to provide a 100% pure offensive analysis."
        poss_heat_help = "Displays the total minutes each team held active, established possession. Purely neutral moments, transitions, or major stoppages are excluded to provide a true reflection of territorial control."
        block_heat_help = "Understanding the 'Team Positioning' Chart:\n\nWhile the first heatmap shows where the team had the ball, this second heatmap shows where the team built their defensive wall when they didn't have the ball.\n\n• Defensive Third (Low Block): The team 'parked the bus.' They retreated deep into their own penalty area to defend.\n• Middle Third (Mid Block): The team held their defensive line near the midfield circle, staying compact.\n• Attacking Third (High Press): The team pushed their defenders aggressively high up the pitch into the opponent's half to trap them."
        mom_help = "A zero-sum measure of Absolute Threat. Only the team controlling the minute scores points. Spikes represent counter-attacks or deep penetration."
        avg_mom_help = "The average Threat Score generated by this team while in control of the match. It evaluates pitch position, attacking tempo, and transitions."

        if st.session_state.view_mode == 'ticker':
            if st.button("⬅️ Back to Match Charts"):
                st.session_state.view_mode = 'dashboard'
                st.rerun()
                
            st.header("⏱️ Minute-by-Minute Live Match Ticker")
            st.markdown("---")
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                
                a_poss = int(row.get('team_a_has_ball', 0)) == 1
                b_poss = int(row.get('team_b_has_ball', 0)) == 1
                zone = str(row['ball_zone']).replace('_', ' ').title()

                if a_poss:
                    narrative = f"**{team_a}** held possession in the **{zone}**, operating at a **{str(row['team_a_attacking_tempo']).replace('_', ' ').title()}** tempo. **{team_b}** defended with a **{str(row['team_b_block_height']).title()} Block** (Pressing Intensity: **{str(row['team_b_pressing_intensity'])}/10**)."
                elif b_poss:
                    narrative = f"**{team_b}** held possession in the **{zone}**, operating at a **{str(row['team_b_attacking_tempo']).replace('_', ' ').title()}** tempo. **{team_a}** defended with a **{str(row['team_a_block_height']).title()} Block** (Pressing Intensity: **{str(row['team_a_pressing_intensity'])}/10**)."
                else:
                    narrative = f"Neutral phase in the **{zone}**."
                    
                st.markdown(f"#### Minute {timestamp}")
                st.write(narrative)
                st.markdown("---")

        else:
            if st.button("⏱️ View Live Match Ticker (Minute-by-Minute Timeline)"):
                st.session_state.view_mode = 'ticker'
                st.rerun()

            st.header("Match Segment Overview")
            
            st.subheader("Global Control")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ta_poss_df = df[df['team_a_has_ball'] == 1]['ball_zone']
                if not ta_poss_df.empty and ta_poss_df.value_counts().iloc[0] >= 2:
                    ta_zone_display = ta_poss_df.value_counts().index[0].replace('_', ' ').title()
                else:
                    ta_zone_display = "None"
                st.metric(label=f"{team_a} Primary Zone", value=ta_zone_display, help=zone_help)
                
            with col2:
                tb_poss_df = df[df['team_b_has_ball'] == 1]['ball_zone']
                if not tb_poss_df.empty and tb_poss_df.value_counts().iloc[0] >= 2:
                    tb_zone_display = tb_poss_df.value_counts().index[0].replace('_', ' ').title()
                else:
                    tb_zone_display = "None"
                st.metric(label=f"{team_b} Primary Zone", value=tb_zone_display, help=zone_help)
                
            with col3:
                st.metric(label=f"{team_a} Time in Attack (min)", value=f"{ta_att_count}", help=att_time_help)
            with col4:
                st.metric(label=f"{team_b} Time in Attack (min)", value=f"{tb_att_count}", help=att_time_help)
                
            st.markdown("---")
            
            st.subheader("Advanced Tactical Metrics")
            col5, col6, col7 = st.columns(3)
            
            ta_press_avg = round(pd.to_numeric(df['team_a_pressing_intensity'], errors='coerce').mean(), 1)
            tb_press_avg = round(pd.to_numeric(df['team_b_pressing_intensity'], errors='coerce').mean(), 1)
            
            block_help = "The primary (most frequent) starting position of the team's defensive wall out of possession."
            
            with col5:
                st.metric(label=f"{team_a} Avg Threat", value=team_a_avg_dom, help=avg_mom_help)
                st.metric(label=f"{team_b} Avg Threat", value=team_b_avg_dom, help=avg_mom_help)
                
            with col6:
                st.metric(label=f"{team_a} Avg Pressing Intensity", value=f"{ta_press_avg} / 10", help=press_help)
                st.metric(label=f"{team_b} Avg Pressing Intensity", value=f"{tb_press_avg} / 10", help=press_help)
                
            with col7:
                st.metric(label=f"{team_a} Primary Block Height", value=ta_block_height.replace('_', ' ').title() + " Block", help=block_help)
                st.metric(label=f"{team_b} Primary Block Height", value=tb_block_height.replace('_', ' ').title() + " Block", help=block_help)
            
            st.markdown("---")
            st.header("Tactical Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Attacking Bias (Geometry)", "Active Possession Distribution", "Team Positioning (Block Height)", "Segment Momentum"])
            
            with tab1:
                st.subheader("Attacking Channels", help=bias_help)
                bias_color_map = {
                    'left_flank': '#66b3ff',      
                    'right_flank': '#99ff99',     
                    'central_channel': '#ff9999'  
                }

                col_pie1, col_pie2 = st.columns(2)
                with col_pie1:
                    st.markdown(f"**{team_a}**")
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    
                    valid_channels = ['left_flank', 'right_flank', 'central_channel']
                    
                    bias_a_df = df[(df['team_a_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
                    bias_a = bias_a_df['team_a_attacking_bias'].dropna()
                    bias_a = bias_a[bias_a.str.lower().str.strip().isin(valid_channels)].value_counts()
                    
                    if not bias_a.empty:
                        clean_idx_a = [str(idx).lower().strip().replace(' ', '_') for idx in bias_a.index]
                        colors_a = [bias_color_map.get(idx, '#cccccc') for idx in clean_idx_a]
                        
                        def make_autopct(values):
                            def my_autopct(pct):
                                total = sum(values)
                                val = int(round(pct*total/100.0))
                                return '{p:.1f}%\n({v:d} times)'.format(p=pct,v=val)
                            return my_autopct

                        ax1.pie(bias_a, labels=bias_a.index.str.replace('_', ' ').str.title(), autopct=make_autopct(bias_a), colors=colors_a)
                        st.pyplot(fig1)
                        
                        buf1 = io.BytesIO()
                        fig1.savefig(buf1, format="png", bbox_inches="tight")
                        st.download_button("💾 Download Chart", buf1.getvalue(), f"{team_a}_attacking_bias.png", "image/png", key="dl_pie_a")
                    else:
                        st.info("No active build-up or attacking sequences logged for this segment.")
                    
                with col_pie2:
                    st.markdown(f"**{team_b}**")
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    
                    bias_b_df = df[(df['team_b_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
                    bias_b = bias_b_df['team_b_attacking_bias'].dropna()
                    bias_b = bias_b[bias_b.str.lower().str.strip().isin(valid_channels)].value_counts()
                    
                    if not bias_b.empty:
                        clean_idx_b = [str(idx).lower().strip().replace(' ', '_') for idx in bias_b.index]
                        colors_b = [bias_color_map.get(idx, '#cccccc') for idx in clean_idx_b]
                        
                        def make_autopct(values):
                            def my_autopct(pct):
                                total = sum(values)
                                val = int(round(pct*total/100.0))
                                return '{p:.1f}%\n({v:d} times)'.format(p=pct,v=val)
                            return my_autopct

                        ax2.pie(bias_b, labels=bias_b.index.str.replace('_', ' ').str.title(), autopct=make_autopct(bias_b), colors=colors_b)
                        st.pyplot(fig2)
                        
                        buf2 = io.BytesIO()
                        fig2.savefig(buf2, format="png", bbox_inches="tight")
                        st.download_button("💾 Download Chart", buf2.getvalue(), f"{team_b}_attacking_bias.png", "image/png", key="dl_pie_b")
                    else:
                        st.info("No active build-up or attacking sequences logged for this segment.")
                    
            with tab2:
                st.subheader("Active Possession Distribution (Total Minutes)", help=poss_heat_help)
                
                zones = ['defensive_third', 'middle_third', 'attacking_third']
                hm_a = [df[(df['team_in_possession'].str.lower() == color_a.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]
                hm_b = [df[(df['team_in_possession'].str.lower() == color_b.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]
                
                heatmap_data = pd.DataFrame([hm_a, hm_b], columns=zones, index=[team_a, team_b])
                
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='g', linewidths=.5, ax=ax3)
                ax3.set_ylabel('') 
                ax3.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax3.get_xticklabels()])
                ax3.set_xlabel('Pitch Zone')
                st.pyplot(fig3)
                
                buf3 = io.BytesIO()
                fig3.savefig(buf3, format="png", bbox_inches="tight")
                st.download_button("💾 Download Heatmap", buf3.getvalue(), "possession_heatmap.png", "image/png", key="dl_heat1")

            with tab3:
                st.subheader("Team Positioning (Defensive Block Height)", help=block_heat_help)
                block_map = {'low': 'defensive_third', 'mid': 'middle_third', 'high': 'attacking_third'}
                
                a_blocks = df['team_a_block_height'].str.lower().map(block_map).value_counts()
                b_blocks = df['team_b_block_height'].str.lower().map(block_map).value_counts()
                
                position_df = pd.DataFrame({team_a: a_blocks, team_b: b_blocks}).fillna(0).T
                
                for col in ['defensive_third', 'middle_third', 'attacking_third']:
                    if col not in position_df.columns:
                        position_df[col] = 0
                        
                position_df = position_df[['defensive_third', 'middle_third', 'attacking_third']]
                
                fig5, ax5 = plt.subplots(figsize=(8, 4))
                sns.heatmap(position_df, annot=True, cmap='Purples', fmt='g', linewidths=.5, ax=ax5)
                ax5.set_ylabel('') 
                ax5.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax5.get_xticklabels()])
                ax5.set_xlabel('Pitch Zone')
                st.pyplot(fig5)
                
                buf5 = io.BytesIO()
                fig5.savefig(buf5, format="png", bbox_inches="tight")
                st.download_button("💾 Download Block Heatmap", buf5.getvalue(), "defensive_block_heatmap.png", "image/png", key="dl_heat2")

            with tab4:
                st.subheader("Match Dominance (Absolute Threat)", help=mom_help)
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                
                x_vals = np.arange(len(df))
                y_vals = df['smoothed_net_momentum']
                
                ax4.fill_between(x_vals, y_vals, 0, where=(y_vals >= 0), color='#d9383a', alpha=0.8, label=f'{team_a}', interpolate=True)
                ax4.fill_between(x_vals, y_vals, 0, where=(y_vals <= 0), color='#333333', alpha=0.8, label=f'{team_b}', interpolate=True)
                
                max_momentum = max(abs(y_vals.max()), abs(y_vals.min()))
                y_lim = max(max_momentum, 1.0) 
                ax4.set_ylim(-y_lim - 1, y_lim + 1)
                
                ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
                
                ticks = np.arange(0, len(df), 5)
                ax4.set_xticks(ticks)
                ax4.set_xticklabels(df['timestamp'].iloc[ticks], rotation=45)

                ax4.set_ylabel('Absolute Attacking Threat')
                ax4.legend()
                ax4.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig4)
                
                buf4 = io.BytesIO()
                fig4.savefig(buf4, format="png", bbox_inches="tight")
                st.download_button("💾 Download Momentum Graph", buf4.getvalue(), "momentum_chart.png", "image/png", key="dl_mom")

            st.markdown("---")
            st.header("🤖 In-Depth AI Diagnostic Report")
            st.markdown(st.session_state.ai_report)
            
            st.download_button(
                label="💾 Download AI Report (Text)",
                data=st.session_state.ai_report.encode('utf-8'),
                file_name=f"{team_a}_vs_{team_b}_AI_Report.md",
                mime='text/markdown'
            )
            
            st.markdown("---")
            st.subheader("💾 Export Raw Data")
            st.write("Download the fully structured, minute-by-minute tactical dataset to run your own models.")
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Tactical Data (CSV)",
                data=csv_data,
                file_name=f"{team_a}_vs_{team_b}_tactical_data.csv",
                mime='text/csv'
            )
