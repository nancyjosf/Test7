import streamlit as st
import cv2
import torch
from model import PullUpCounter
import sys
sys.path.append('C:\\Users\\Nancy\\Downloads\\Test6\\testing.py')
from testing import ExerciseClassifierApp

# Step 1: Create a blank white homepage
st.set_page_config(page_title="Gym Assistant", page_icon="🏋️‍♂️", layout="centered")

# Creating exercise menu
st.sidebar.title("Exercise Menu")

selected_exercise = st.sidebar.selectbox(
    "Exercise Options:",
    ["", "Deadlift", "Push-Ups", "Bicep Curl", "Squat", "Pull-Ups", "Lunges"],
    key="exercise",
)

st.sidebar.write("")
st.sidebar.markdown("You can select various exercises from the above menu.")

if selected_exercise == "":
    st.title("Welcome to AI Gym Assistant  🏋️‍")

# Add exercise-specific content here
if selected_exercise == "Deadlift":
    st.sidebar.markdown("**Deadlift** is a great exercise for building strength.")
    st.sidebar.write("**Step 1: Set Up**.\n1. Stand on your feet shoulder-width apart, toes pointing forward.\n2. The barbell should be over the middle of your feet.")
    st.sidebar.write("**Step 2: Grip**")
    st.sidebar.write("3. Bend at the hips and knees to grasp the barbell with an overhand grip (palms facing you) or mixed grip (one palm facing you, one away).")
    st.sidebar.write("**Step 3: Stance**")
    st.sidebar.write("4. Your hands should be just outside your knees.\n5. Keep your back straight, chest up, and shoulders back.")
    st.sidebar.write("**Step 5: Lowering**")
    st.sidebar.write("9. Reverse the movement, pushing your hips back first.\n10. Lower the barbell with control, keeping it close to your body.")

if selected_exercise == "Push-Ups":
    st.sidebar.markdown("**Push-Ups** are a versatile exercise that targets multiple muscle groups, including the chest, shoulders, and triceps.")
    st.sidebar.write("\nHere are some tips for performing the Push-Ups:")
    st.sidebar.write("Step 1: Starting Position")
    st.sidebar.write("1. Begin in a plank position, hands shoulder-width apart, arms fully extended, and body straight from head to heels.")
    st.sidebar.write("**Step 2: Descent**")
    st.sidebar.write("2. Lower your body by bending your elbows, keeping them close to your sides.\n3. Lower until your chest nearly touches the ground, or as far as your strength allows.")
    st.sidebar.write("**Step 3: Pushing Up**")
    st.sidebar.write("4. Push through your palms to straighten your arms, returning to the starting position.")

if selected_exercise == "Lunges":
    st.sidebar.markdown("**Lunges** are a great exercise for strengthening your legs and improving overall lower body strength.")
    st.sidebar.write("**Step 1: Starting Position**")
    st.sidebar.write("1. Stand tall with your feet hip-width apart. Place your hands on your hips or hold weights by your sides for added resistance.")
    st.sidebar.write("**Step 2: Lunge Forward**")
    st.sidebar.write("2. Take a step forward with one leg, lowering your hips until both knees are bent at about a 90-degree angle.\n3. Ensure your front knee is directly above your ankle and your other knee hovers just above the ground.")
    st.sidebar.write("**Step 3: Return to Starting Position**")
    st.sidebar.write("4. Push through your front heel to return to the starting position.\n5. Repeat on the other side.")

    st.write("**Lunge Detection**")
    
   
    model_path = 'exercise_classifier.pth'
    label_encoder_path = 'label_encoder.npy'
    input_size = 66  # 33 landmarks * 2 (x, y)
    num_classes = 4  # Adjust based on your dataset

    lunge_counter = ExerciseClassifierApp(model_path, label_encoder_path, input_size, num_classes)
    start_detection = st.button("Start Lunge Detection")
    stop_detection = st.button("Stop Lunge Detection")
    stframe = st.empty()

    if start_detection:
        cap = cv2.VideoCapture(0)
        run_detection = True

        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            # Process and classify the frame
            landmarks = lunge_counter.process_frame(frame)
            if landmarks is not None:
                classification_result, is_valid = lunge_counter.classify_frame(frame)
                if is_valid:
                    lunge_counter.correct_predictions += 1
                accuracy = lunge_counter.correct_predictions / (lunge_counter.frame_count + 1)
                lunge_counter.display_frame(frame, classification_result, accuracy)
            else:
                cv2.putText(frame, "No valid pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

            if stop_detection or cv2.waitKey(1) & 0xFF == ord('q'):
                run_detection = False

            lunge_counter.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

if selected_exercise == "Pull-Ups":
    st.sidebar.markdown("**Pull-Ups** are a challenging exercise for building upper body strength.")
    st.write("**Pull-Up Detection**")
    
    pullup_counter = PullUpCounter()
    start_detection = st.button("Start Pull-Up Detection")
    stop_detection = st.button("Stop Pull-Up Detection")
    stframe = st.empty()
    
    if start_detection:
        cap = cv2.VideoCapture(0)
        run_detection = True
        
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            frame = pullup_counter.start_detection(frame)
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            if stop_detection or cv2.waitKey(1) & 0xFF == ord('q'):
                run_detection = False
        
        cap.release()
        cv2.destroyAllWindows()
