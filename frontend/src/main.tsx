import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// StrictMode removed: it causes double-invocation of useEffect in dev,
// which triggers two Ollama/RAG calls every time a report is generated.
createRoot(document.getElementById('root')!).render(
  <App />
);
