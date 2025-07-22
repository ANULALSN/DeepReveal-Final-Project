import React, { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Loader, UploadCloud, Image as ImageIcon, AlertCircle, LogIn, UserPlus, Home, LogOut, User, Cpu, Sun, Moon } from 'lucide-react';
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged 
} from 'firebase/auth';

// --- 1. Firebase Configuration ---
// PASTE YOUR FIREBASE CONFIG OBJECT HERE
const firebaseConfig = {
  apiKey: "AIzaSyCaCepaDwDcNOZ6Z2rxN2N_z3vbNbwqBkg",
  authDomain: "deepreveal-final-project.firebaseapp.com",
  projectId: "deepreveal-final-project",
  storageBucket: "deepreveal-final-project.firebasestorage.app",
  messagingSenderId: "490175026389",
  appId: "1:490175026389:web:fa78392ecf46b03e263e5c",
  measurementId: "G-6YGG3FL6V2"
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);

// --- 2. API Configuration ---
const API_URL = 'http://127.0.0.1:8000/predict/';

// --- 3. Page Components ---

// --- LOGIN PAGE ---
const LoginPage = ({ setPage }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col justify-center items-center p-4 transition-colors duration-300">
      <div className="max-w-md w-full mx-auto bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-xl">
        <h1 className="text-4xl font-bold text-center text-gray-800 dark:text-gray-100 mb-2">Welcome Back</h1>
        <p className="text-center text-gray-500 dark:text-gray-400 mb-6">Sign in to continue to DeepReveal</p>
        <form onSubmit={handleLogin}>
          {error && <p className="bg-red-100 text-red-700 p-3 rounded-lg mb-4">{error}</p>}
          <div className="mb-4">
            <label className="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2" htmlFor="email">Email Address</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 dark:text-white" required />
          </div>
          <div className="mb-6">
            <label className="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2" htmlFor="password">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 dark:text-white" required />
          </div>
          <button type="submit" disabled={loading} className="w-full bg-blue-600 text-white font-bold py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center">
            {loading ? <Loader className="animate-spin" /> : <><LogIn className="mr-2"/> Sign In</>}
          </button>
        </form>
        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-6">
          Don't have an account? <span onClick={() => setPage('signup')} className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer font-semibold">Sign Up</span>
        </p>
      </div>
    </div>
  );
};

// --- SIGNUP PAGE ---
const SignupPage = ({ setPage }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
  
    const handleSignup = async (e) => {
      e.preventDefault();
      setLoading(true);
      setError('');
      try {
        await createUserWithEmailAndPassword(auth, email, password);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col justify-center items-center p-4 transition-colors duration-300">
      <div className="max-w-md w-full mx-auto bg-white dark:bg-gray-800 p-8 rounded-2xl shadow-xl">
        <h1 className="text-4xl font-bold text-center text-gray-800 dark:text-gray-100 mb-2">Create Account</h1>
        <p className="text-center text-gray-500 dark:text-gray-400 mb-6">Get started with DeepReveal</p>
        <form onSubmit={handleSignup}>
          {error && <p className="bg-red-100 text-red-700 p-3 rounded-lg mb-4">{error}</p>}
          <div className="mb-4">
            <label className="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2" htmlFor="email">Email Address</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 dark:text-white" required />
          </div>
          <div className="mb-6">
            <label className="block text-gray-700 dark:text-gray-300 text-sm font-bold mb-2" htmlFor="password">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-3 py-2 border dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 dark:text-white" required />
          </div>
          <button type="submit" disabled={loading} className="w-full bg-blue-600 text-white font-bold py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center">
            {loading ? <Loader className="animate-spin" /> : <><UserPlus className="mr-2"/> Sign Up</>}
          </button>
        </form>
        <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-6">
          Already have an account? <span onClick={() => setPage('login')} className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer font-semibold">Sign In</span>
        </p>
      </div>
    </div>
  );
};

// --- NAVBAR ---
const Navbar = ({ user, setPage, theme, toggleTheme }) => {
    const handleLogout = async () => {
        await signOut(auth);
    };

    return (
        <header className="bg-white dark:bg-gray-800 shadow-md w-full transition-colors duration-300">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center">
                        <Cpu className="h-8 w-8 text-blue-600" />
                        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100 ml-2">DeepReveal</h1>
                    </div>
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center">
                            <User className="h-5 w-5 text-gray-500 dark:text-gray-300 mr-2"/>
                            <span className="text-gray-700 dark:text-gray-200 font-medium">{user.email}</span>
                        </div>
                        <button onClick={toggleTheme} className="p-2 rounded-full text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                            {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
                        </button>
                        <button onClick={handleLogout} className="flex items-center bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600">
                            <LogOut className="h-5 w-5 mr-2"/> Logout
                        </button>
                    </div>
                </div>
            </nav>
        </header>
    );
};

// --- HOME PAGE ---
const HomePage = ({ user, setPage, theme, toggleTheme }) => (
  <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
    <Navbar user={user} setPage={setPage} theme={theme} toggleTheme={toggleTheme} />
    <main className="max-w-4xl mx-auto mt-12 p-8">
        <div className="bg-white dark:bg-gray-800 p-10 rounded-2xl shadow-xl text-center">
            <h2 className="text-4xl font-bold text-gray-800 dark:text-gray-100">What is a Deepfake?</h2>
            <p className="text-lg text-gray-600 dark:text-gray-300 mt-4 leading-relaxed">
                A deepfake is a form of synthetic media where a person in an existing image or video is replaced with someone else's likeness. Leveraging powerful techniques from artificial intelligence and machine learning, deepfakes can produce highly realistic and convincing fraudulent content. This technology poses significant challenges related to misinformation, security, and identity theft.
            </p>
            <p className="text-lg text-gray-600 dark:text-gray-300 mt-4 leading-relaxed">
                The **DeepReveal** project is a lightweight, explainable framework designed to combat this threat by accurately detecting and localizing manipulated regions in images.
            </p>
            <button onClick={() => setPage('app')} className="mt-8 bg-blue-600 text-white font-bold py-4 px-8 rounded-lg text-xl hover:bg-blue-700 transition-transform transform hover:scale-105">
                Try DeepReveal Now
            </button>
        </div>
    </main>
  </div>
);

// --- DEEPREVEAL APP PAGE ---
const DeepRevealApp = ({ user, setPage, theme, toggleTheme }) => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const onDrop = useCallback(acceptedFiles => {
        if (acceptedFiles && acceptedFiles[0]) {
            setFile(acceptedFiles[0]);
            setPreview(URL.createObjectURL(acceptedFiles[0]));
            setResult(null);
            setError(null);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'image/*': ['.jpeg', '.jpg', '.png'] }, multiple: false });

    const handleAnalyze = async () => {
        if (!file || !auth.currentUser) {
            setError('Please upload an image and ensure you are logged in.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);

        const token = await auth.currentUser.getIdToken();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData,
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
            }
            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(`API Error: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleReset = () => { setFile(null); setPreview(null); setResult(null); setError(null); setIsLoading(false); };
    
    const ResultCard = ({ label, value, isFake }) => ( <div className={`p-4 rounded-lg ${isFake ? 'bg-red-100 dark:bg-red-900/40' : 'bg-green-100 dark:bg-green-900/40'}`}><p className="text-sm font-medium text-gray-500 dark:text-gray-400">{label}</p><p className={`text-2xl font-bold ${isFake ? 'text-red-700 dark:text-red-400' : 'text-green-700 dark:text-green-400'}`}>{value}</p></div> );
    const ConfidenceMeter = ({ confidence }) => { const fakeConfidence = parseFloat(confidence.FAKE) * 100; const realConfidence = parseFloat(confidence.REAL) * 100; return (<div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-6 my-4 overflow-hidden flex"><div className="bg-red-500 h-6 flex items-center justify-center text-white text-xs font-bold" style={{ width: `${fakeConfidence}%` }}>{fakeConfidence > 10 && `FAKE ${fakeConfidence.toFixed(1)}%`}</div><div className="bg-green-500 h-6 flex items-center justify-center text-white text-xs font-bold" style={{ width: `${realConfidence}%` }}>{realConfidence > 10 && `REAL ${realConfidence.toFixed(1)}%`}</div></div>);};

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
            <Navbar user={user} setPage={setPage} theme={theme} toggleTheme={toggleTheme} />
            <main className="w-full max-w-4xl mx-auto p-4 mt-8">
                <button onClick={() => setPage('home')} className="mb-4 flex items-center text-blue-600 dark:text-blue-400 hover:underline">
                    <Home className="mr-2 h-5 w-5"/> Back to Home
                </button>
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 md:p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="flex flex-col justify-between">
                        <div>
                            <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-200 mb-4">Upload Image</h2>
                            <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors duration-300 ${isDragActive ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'}`}>
                                <input {...getInputProps()} />
                                {preview ? <img src={preview} alt="Preview" className="mx-auto max-h-48 rounded-lg shadow-sm" /> : <div className="flex flex-col items-center text-gray-500 dark:text-gray-400"><UploadCloud className="w-12 h-12 mb-2" /><p>Drag & drop an image here</p></div>}
                            </div>
                        </div>
                        <div className="mt-6 flex flex-col space-y-3">
                            <button onClick={handleAnalyze} disabled={!file || isLoading} className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center text-lg shadow-md"><Loader className={`animate-spin mr-2 ${!isLoading && 'hidden'}`} /> Analyze Image</button>
                            <button onClick={handleReset} className="w-full bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 font-bold py-2 px-4 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-500">Reset</button>
                        </div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-xl p-6 flex flex-col items-center justify-center min-h-[300px]">
                        <h2 className="text-2xl font-semibold text-gray-700 dark:text-gray-200 mb-4 self-start">Analysis Result</h2>
                        {isLoading && <div className="text-center text-gray-600 dark:text-gray-300"><Loader className="w-16 h-16 animate-spin text-blue-500 mx-auto" /><p className="mt-4 text-lg">Processing...</p></div>}
                        {error && <div className="text-center text-red-600 bg-red-100 p-4 rounded-lg"><AlertCircle className="w-12 h-12 mx-auto mb-2" /><p className="font-bold">An Error Occurred</p><p className="text-sm">{error}</p></div>}
                        {!isLoading && !error && result && <div className="w-full text-center"><img src={result.result_image} alt="Analysis Result" className="rounded-lg shadow-lg mx-auto mb-4 w-full" /><ConfidenceMeter confidence={result.confidence} /><div className="grid grid-cols-2 gap-4 mt-4"><ResultCard label="Prediction" value={result.prediction} isFake={result.prediction === 'FAKE'} /><ResultCard label="Confidence" value={`${(Math.max(parseFloat(result.confidence.FAKE), parseFloat(result.confidence.REAL)) * 100).toFixed(2)}%`} isFake={result.prediction === 'FAKE'} /></div></div>}
                        {!isLoading && !error && !result && <div className="text-center text-gray-400"><ImageIcon className="w-16 h-16 mx-auto mb-4" /><p>Results will appear here.</p></div>}
                    </div>
                </div>
            </main>
        </div>
    );
};


// --- 4. Main App Component (Router) ---
const App = () => {
  const [page, setPage] = useState('login');
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState(localStorage.getItem('deepreveal-theme') || 'light');

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('deepreveal-theme', theme);
  }, [theme]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        setPage('home');
      } else {
        setUser(null);
        setPage('login');
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
            <Loader className="w-24 h-24 animate-spin text-blue-600"/>
        </div>
    );
  }

  // Simple Router Logic
  switch (page) {
    case 'signup':
      return <SignupPage setPage={setPage} />;
    case 'home':
      return <HomePage user={user} setPage={setPage} theme={theme} toggleTheme={toggleTheme} />;
    case 'app':
      return <DeepRevealApp user={user} setPage={setPage} theme={theme} toggleTheme={toggleTheme} />;
    case 'login':
    default:
      return <LoginPage setPage={setPage} />;
  }
};

export default App;
