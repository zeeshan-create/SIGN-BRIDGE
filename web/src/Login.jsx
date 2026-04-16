import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { ArrowRight, Lock, Mail, User } from 'lucide-react';
import { GoogleLogin } from '@react-oauth/google';

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  // Dynamic API Base URL
  const API_BASE = import.meta.env.VITE_SERVER_URL || `http://${window.location.hostname}:5000`;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
    const url = `${API_BASE}${endpoint}`;
    
    console.log(`[Auth] Attempting ${isLogin ? 'Login' : 'Signup'} at: ${url}`);
    
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      const data = await res.json();
      console.log('[Auth] Response:', data);

      if (!res.ok) throw new Error(data.message || 'Authentication failed');
      
      login(data.user, data.token);
      navigate('/dashboard');
    } catch (err) {
      console.error('[Auth] Connection Error:', err);
      setError(err.message === 'Failed to fetch' 
        ? 'Could not connect to the server. Please ensure the backend is running on port 5000.' 
        : err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');
    const url = `${API_BASE}/api/auth/google`;
    
    console.log('[Auth] Verifying Google token at:', url);
    
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: credentialResponse.credential })
      });
      
      const data = await res.json();
      console.log('[Auth] Google Response:', data);

      if (!res.ok) throw new Error(data.message || 'Google authentication failed');
      
      login(data.user, data.token);
      navigate('/dashboard');
    } catch (err) {
      console.error('[Auth] Google Connection Error:', err);
      setError(err.message === 'Failed to fetch' 
        ? 'Could not connect to the server for Google verification. Ensure backend is running.' 
        : err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.12, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[var(--bg-primary)] flex flex-col items-center justify-center px-6 py-12 transition-colors duration-300"
    >
      <div className="w-full max-w-[440px] space-y-12">
        <header className="text-center space-y-4">
          <div className="flex flex-col items-center gap-6 mb-8">
             <Link to="/" className="hover:opacity-80 transition-opacity">
               <img src="/logo.png" className="w-32 object-contain" alt="SignBridge" />
             </Link>
             <div className="w-24 h-px bg-gradient-to-r from-transparent via-[#10b981]/20 to-transparent" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900">
            {isLogin ? 'Welcome back.' : 'Create your account.'}
          </h1>
          <p className="text-gray-500 font-medium">
            Where Signs Become Voice.
          </p>
        </header>

        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-100 p-4 rounded-xl text-red-600 text-sm font-bold text-center"
          >
            {error}
          </motion.div>
        )}

        <div className="space-y-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {!isLogin && (
              <div className="space-y-2">
                <label className="text-[0.7rem] font-bold uppercase tracking-widest text-gray-400">Full Name</label>
                <div className="relative group">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300 group-focus-within:text-emerald-500 transition-colors" size={18} />
                  <input 
                    required
                    type="text"
                    className="w-full bg-white border border-gray-200 rounded-xl p-4 pl-12 text-sm font-medium focus:ring-1 focus:ring-emerald-500 outline-none transition-all shadow-sm"
                    placeholder="Zeeshan Create"
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-[0.7rem] font-bold uppercase tracking-widest text-gray-400">Email Address</label>
              <div className="relative group">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300 group-focus-within:text-emerald-500 transition-colors" size={18} />
                <input 
                  required
                  type="email"
                  className="w-full bg-white border border-gray-200 rounded-xl p-4 pl-12 text-sm font-medium focus:ring-1 focus:ring-emerald-500 outline-none transition-all shadow-sm"
                  placeholder="you@example.com"
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-[0.7rem] font-bold uppercase tracking-widest text-gray-400">Password</label>
              <div className="relative group">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300 group-focus-within:text-emerald-500 transition-colors" size={18} />
                <input 
                  required
                  type="password"
                  className="w-full bg-white border border-gray-200 rounded-xl p-4 pl-12 text-sm font-medium focus:ring-1 focus:ring-emerald-500 outline-none transition-all shadow-sm"
                  placeholder="••••••••"
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                />
              </div>
            </div>

            <motion.button 
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.02, boxShadow: "0 0 25px rgba(16, 185, 129, 0.4)" }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-[#10b981] text-white py-4 rounded-xl font-bold transition-all shadow-xl shadow-green-100 hover:bg-[#059669] active:scale-[0.98] disabled:opacity-50 flex items-center justify-center gap-3"
            >
              {loading ? (isLogin ? 'Signing In...' : 'Creating...') : (isLogin ? 'Sign In' : 'Create Account')}
              {!loading && <ArrowRight size={18} />}
            </motion.button>
          </form>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-100"></div>
            </div>
            <div className="relative flex justify-center text-[0.6rem] uppercase tracking-widest">
              <span className="bg-[var(--bg-primary)] text-[var(--text-tertiary)] font-bold">Or continue with</span>
            </div>
          </div>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => setError('Google Login Failed')}
              useOneTap
              theme="outline"
              shape="pill"
              text={isLogin ? "signin_with" : "signup_with"}
              width="440px"
            />
          </div>
        </div>

        <footer className="text-center">
          <button 
            onClick={() => setIsLogin(!isLogin)}
            className="text-gray-400 text-sm font-bold hover:text-emerald-600 transition-colors underline decoration-gray-200"
          >
            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Log in"}
          </button>
        </footer>
      </div>
    </motion.div>
  );
}
