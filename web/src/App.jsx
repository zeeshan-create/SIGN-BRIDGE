import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { io } from 'socket.io-client';

// Global Framer Motion performance optimizations
motion.defaults.layoutAnimations = false;
motion.defaults.reduceMotion = 'user';

import Header from './Header';
import Home from './Home';
import Login from './Login';
import Dashboard from './Dashboard';
import Profile from './Profile';
import Admin from './Admin';
import Converter from './Converter';
import About from './About';
import FAQ from './FAQ';
import Contact from './Contact';
import Blog from './Blog';
import { AuthProvider, useAuth } from './AuthContext';
import { ThemeProvider } from './ThemeContext';
import ErrorBoundary from './ErrorBoundary';

// Connect to the Node.js backend
const socket = io(import.meta.env.VITE_SERVER_URL || 'http://localhost:5000');

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (!user) return <Login />;
  return children;
}

function AdminProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (!user || user.role !== 'admin') return <Dashboard socket={socket} />;
  return children;
}

function AnimatedRoutes() {
  const location = useLocation();
  
  // Smooth scroll to top on every route change
  React.useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  }, [location.pathname]);

  const pageVariants = {
    initial: {
      opacity: 0,
      y: 8,
    },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.3,
        ease: [0.25, 0.1, 0.25, 1],
      }
    },
    exit: {
      opacity: 0,
      y: -4,
      transition: {
        duration: 0.2,
        ease: [0.25, 0.1, 0.25, 1],
      }
    }
  };

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={location.pathname}
        initial="initial"
        animate="animate"
        exit="exit"
        variants={pageVariants}
        className="min-h-full"
      >
        <Routes location={location}>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/about" element={<About />} />
        <Route path="/faq" element={<FAQ socket={socket} />} />
        <Route path="/contact" element={<Contact socket={socket} />} />
        <Route path="/blog" element={<Blog />} />

        <Route path="/converter" element={<Converter socket={socket} />} />
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <Dashboard socket={socket} />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/profile" 
          element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/admin" 
          element={
            <AdminProtectedRoute>
              <Admin />
            </AdminProtectedRoute>
          } 
        />
      </Routes>
      </motion.div>
    </AnimatePresence>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <Router>
            <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] transition-colors duration-300">
              <Header />
              <AnimatedRoutes />
            </div>
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
