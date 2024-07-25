import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
from tkinter import messagebox

# Load the model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Prediction function
def predict_image(model, img):
    img = img.resize((256, 256)) 
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    predictions = model.predict(img_array)
    
    # Debugging: Print raw predictions
    print(f"Raw predictions: {predictions}")

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    # Debugging: Print predicted class and confidence
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}%")
    
    return predicted_class, confidence

# Select image function
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path).convert('RGB')
            img.thumbnail((256, 256))  
            img_tk = ImageTk.PhotoImage(img)
            l1.configure(image=img_tk)
            l1.image = img_tk

            # Predict
            predicted_class, confidence = predict_image(model, img)
            result_text.set(f"Prediction: {predicted_class}\nConfidence: {confidence}%")
            result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            # Place the image label after prediction
            l1.place(relx=0.5, rely=0.75, anchor=tk.CENTER)
        except Exception as e:
            messagebox.showerror("Error", str(e))

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.geometry("1920x1080")
app.title("SpudSage.in")

img1 = ImageTk.PhotoImage(Image.open("img1.jpg"))
l1 = ctk.CTkLabel(master=app, image=img1)
l1.pack()




frame = ctk.CTkFrame(master=app, width=670, height=800,fg_color="black")
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


logo_image = Image.open("logo1.png") 
logo_image = logo_image.resize((670, 420), Image.Resampling.LANCZOS)
logo_image_tk = ImageTk.PhotoImage(logo_image)

# Label to display the logo at the top
logo_label = tk.Label(master=frame, image=logo_image_tk)
logo_label.place(relx=0.5, rely=0.10, anchor=tk.CENTER)

l2 = ctk.CTkLabel(master=frame, text="Welcome to SpudSage", font=('Arial', 26, 'bold'))
l2.place(relx=0.5, rely=0.30, anchor=tk.CENTER)

l3 = ctk.CTkLabel(master=frame, text="Check the condition of your potato leaves with just a few clicks!", font=('Arial', 22))
l3.place(relx=0.5, rely=0.37, anchor=tk.CENTER)

button1 = ctk.CTkButton(master=frame, text="Select Image and Predict", width=300, corner_radius=10, command=select_image, font=('Arial', 18))
button1.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

# Label to display prediction results, initially hidden
result_text = tk.StringVar()
result_label = tk.Label(master=frame, textvariable=result_text, font=('Arial', 18), fg="white", bg="black")
result_label.place(relx=0.5, rely=0.70, anchor=tk.CENTER)  


l1 = ctk.CTkLabel(master=frame)

app.mainloop()
