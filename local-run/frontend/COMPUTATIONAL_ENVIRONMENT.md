# Computational Environment Description

This project, **Flu Guardian (流感卫士)**, is a full-stack web application designed to run in a modern web browser environment with a Node.js backend for orchestration.

## 1. Client-Side (Frontend)
The frontend is the primary execution environment for real-time monitoring and machine learning tasks.

- **Runtime**: Modern Web Browser (Chrome 90+, Edge 90+, Safari 14+)
- **Framework**: React 18 with TypeScript
- **Machine Learning Engine**: TensorFlow.js (Web-GL accelerated)
- **Audio Processing**: Web Audio API (for real-time spectrogram and feature extraction)
- **Hardware Requirements**:
  - Minimum 4GB RAM
  - Integrated Microphone
  - GPU with WebGL 2.0 support (recommended for faster ML inference/training)

## 2. Server-Side (Backend)
The backend handles authentication, data persistence, and high-level AI reasoning via the Gemini API.

- **Runtime**: Node.js 18+
- **API Access**: Google Gemini API (requires `GEMINI_API_KEY`)
- **Database**: Firebase Firestore (NoSQL)
- **Hosting**: Cloud Run (Containerized environment)

## 3. Machine Learning Workflow
- **Training**: Performed client-side using TensorFlow.js. It utilizes Transfer Learning to adapt a pre-trained voiceprint model to specific student voice characteristics.
- **Inference**: 
  - **Local Inference**: TensorFlow.js runs YAMNet-based cough detection and voiceprint matching in real-time on the browser thread.
  - **Remote Inference**: High-level epidemiological reasoning and risk assessment are performed via the Gemini-3-Flash model.

## 4. Dependencies
Key libraries required for the environment:
- `@tensorflow/tfjs`: Core ML engine
- `@google/genai`: Gemini API SDK
- `react-konva`: Spatial topology rendering
- `recharts`: Data visualization
- `motion`: UI animations
