# Golf Ball Detection AI

A modern web application that uses AI to detect golf balls in images. Built with React, FastAPI, and YOLO object detection.

## Features

- 🎯 Accurate golf ball detection using YOLOv8
- 🖼️ Modern, responsive UI with real-time preview
- 🚀 Fast processing with Python backend
- 💫 Smooth animations and transitions
- 📱 Mobile-friendly design

## Tech Stack

### Frontend
- React with TypeScript
- Tailwind CSS for styling
- Axios for API calls
- Heroicons for icons

### Backend
- FastAPI
- Python 3.10+
- Ultralytics YOLOv8
- Pillow for image processing

## Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 16 or higher
- YOLO model file (best.pt) trained for golf ball detection

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained model file (best.pt) in the `backend/model/` directory

4. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

## Usage

1. Open the application in your browser (default: http://localhost:5173)
2. Click "Choose Image" to select a golf course image
3. Click "Detect Golf Balls" to process the image
4. View the results with detected golf balls highlighted

## Project Structure

```
├── backend/
│   ├── model/          # YOLO model directory
│   ├── main.py         # FastAPI application
│   └── model_loader.py # Model loading utilities
│
└── frontend/
    ├── src/
    │   ├── App.tsx    # Main application component
    │   └── index.css  # Global styles
    └── public/        # Static assets
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this project for your own purposes.

## Acknowledgments

- YOLOv8 by Ultralytics
- React and FastAPI communities
- Tailwind CSS team 