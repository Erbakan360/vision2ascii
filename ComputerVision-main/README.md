# Vision2ASCII Project

📌 Project Overview:
--------------------------------------------------
Vision2ASCII is a computer vision-based application that converts images and live webcam input into ASCII art. Users can upload an image or capture real-time video, apply filters like grayscale or negative, and use edge detection techniques such as Canny and Sobel. The final ASCII representation can be copied or saved in .txt format or as an image screenshot.

The project showcases key computer vision concepts like:
- Intensity mapping
- Filtering
- Edge detection
- Real-time image processing

🛠 Features:
--------------------------------------------------
- Upload an image or use live webcam feed
- Apply filters (Grayscale, Negative)
- Perform edge detection (Canny, Sobel)
- ASCII art generation from images
- Save output as .txt or image screenshot
- Simple graphical user interface (GUI)

👥 Group Members:
--------------------------------------------------
• Erbakan Ahmad (12306435)  
• Jamal Dassrath (22301035)  
• Love - (12306406)  
• Muhammad Ahtisham Bhatti (22301502)

💻 Technologies Used:
--------------------------------------------------
- Python
- OpenCV
- NumPy
- Tkinter (or other GUI framework)
- PIL (Pillow)

📁 Project Structure:
--------------------------------------------------
Vision2ASCII/
├── main.py                 # Entry point of the application
├── gui.py                  # Handles GUI elements
├── ascii_converter.py      # Core logic for image-to-ASCII conversion
├── filters.py              # Filter and edge detection operations
├── utils.py                # Helper functions
├── assets/                 # Icons, sample images, etc.
├── output/                 # Saved .txt or image screenshots
└── README.txt              # Project description and documentation

📦 Setup Instructions:
--------------------------------------------------
0. Setup python Virtual Enviroment (Recomended)

   ```sh
   python3 -m venv .venv
   ```

   ```sh
   source .venv/bin/activate
   ```
 

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Run the application:
   ```sh
   streamlit run main.py
   ```

📜 License:
--------------------------------------------------
This project is for educational purposes.

📧 Contact:
--------------------------------------------------
For any questions or issues, please contact any of the group members.

==================================================

