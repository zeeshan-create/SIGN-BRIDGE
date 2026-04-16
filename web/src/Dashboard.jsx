import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import {
  Search, Bell, Settings, ArrowRight, Play, RefreshCcw, Camera, Target,
  History as HistoryIcon, User, Terminal, Zap, ChevronRight, Waves, Hash,
  MessageSquare, Users, Coffee, MapPin, Download, Power, Video, VideoOff,
  BrainCircuit, LayoutGrid, Sparkles, AlertCircle, ShieldCheck, ChevronLeft,
  Eye, Activity, Database, Radio, Layers, Network, Fingerprint, Upload, Sun, Moon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from './AuthContext';
import { useTheme } from './ThemeContext';
import Profile from './Profile';
import SettingsView from './Settings';
import { Link } from 'react-router-dom';
import CategoriesView from './CategoriesView';
import CommunityView from './CommunityView';
import axios from 'axios';

// --- Optimized Scanline Overlay ---
const ScanLine = ({ isVisible }) => (
  <motion.div 
    initial={{ y: '-10%' }}
    animate={isVisible ? { y: '120%' } : { y: '-10%' }}
    transition={{ duration: 1.8, repeat: isVisible ? Infinity : 0, ease: "linear" }}
    className="absolute left-0 right-0 h-[2px] bg-[#16A34A]/20 blur-[1px] z-10 pointer-events-none transform translate-z-0 backface-hidden"
  />
);

// --- Pure SVG Waveform Component ---
const WaveformChart = () => {
  const points = [10, 40, 30, 60, 45, 80, 55, 90, 70, 95, 85, 100];
  const width = 1000;
  const height = 200;
  
  const pathData = points.reduce((acc, point, i, a) => {
    const x = (i / (a.length - 1)) * width;
    const y = height - (point / 100) * height;
    if (i === 0) return `M ${x},${y}`;
    const prevX = ((i - 1) / (a.length - 1)) * width;
    const prevY = height - (a[i - 1] / 100) * height;
    const cpX = (prevX + x) / 2;
    return `${acc} C ${cpX},${prevY} ${cpX},${y} ${x},${y}`;
  }, "");

  const fillPath = `${pathData} L ${width},${height} L 0,${height} Z`;

  return (
    <div className="w-full h-64 relative bg-[var(--bg-primary)] overflow-hidden">
       <div className="absolute inset-0 flex flex-col justify-between opacity-5 pointer-events-none">
          {[...Array(5)].map((_, i) => <div key={i} className="h-px bg-[#16A34A] w-full" />)}
       </div>
       <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible group">
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#16A34A" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#16A34A" stopOpacity="0" />
            </linearGradient>
          </defs>
          <motion.path 
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
            d={fillPath} 
            fill="url(#gradient)" 
          />
          <motion.path 
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
            d={pathData} 
            fill="none" 
            stroke="#16A34A" 
            strokeWidth="3" 
            strokeLinecap="round"
          />
          {points.map((p, i) => (
             <motion.circle 
               key={i}
               initial={{ scale: 0 }}
               animate={{ scale: 1 }}
               transition={{ delay: 0.06 + (i * 0.01) }}
               cx={(i / (points.length - 1)) * width} 
               cy={height - (p / 100) * height} 
               r="4" 
               fill="white" 
               stroke="#16A34A" 
               strokeWidth="2"
               className="hover:r-6 cursor-pointer transition-all"
             />
          ))}
       </svg>
    </div>
  );
};

const CategoryGrid = () => {
  const categories = [
    { name: 'Greetings', icon: Waves, progress: 92, count: '48/52 Signs' },
    { name: 'Numbers 1-50', icon: Hash, progress: 85, count: '170/200' },
    { name: 'Common Phrases', icon: MessageSquare, progress: 67, count: '134/200' },
    { name: 'Family & People', icon: Users, progress: 74, count: '74/100' },
    { name: 'Food & Drink', icon: Coffee, progress: 55, count: '110/200' },
    { name: 'Places', icon: MapPin, progress: 41, count: '41/100' },
  ];
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
       {categories.map((cat, i) => (
         <motion.div 
           initial={{ opacity: 0, y: 10 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ delay: i * 0.01 }}
           key={cat.name} 
           className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-8 shadow-[0_20px_40px_-15px_rgba(22,163,74,0.05)] hover:shadow-[0_40px_80px_-20px_rgba(22,163,74,0.1)] transition-all group cursor-pointer"
         >
            <div className="flex justify-between items-start mb-8">
               <div className="w-14 h-14 bg-[var(--bg-tertiary)] rounded-2xl flex items-center justify-center text-[#16A34A] group-hover:bg-[#16A34A] group-hover:text-white transition-all shadow-inner">
                  <cat.icon size={26} />
               </div>
               <ChevronRight size={18} className="text-[#BBF7D0] group-hover:text-[#16A34A] transition-colors" />
            </div>
            <h4 className="font-black text-[var(--text-primary)] text-lg mb-1 tracking-tight">{cat.name}</h4>
            <div className="flex justify-between items-end mb-4">
               <span className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em]">{cat.count}</span>
               <span className="text-sm font-black text-[#16A34A]">{cat.progress}%</span>
            </div>
            <div className="h-2.5 bg-[var(--bg-tertiary)] rounded-full overflow-hidden border border-[var(--border-color)]/20">
               <motion.div 
                 initial={{ width: 0 }}
                 animate={{ width: `${cat.progress}%` }}
                 transition={{ duration: 0.3, delay: 0.02 + (i * 0.02) }}
                 className="h-full bg-gradient-to-r from-[#16A34A] to-[#4ADE80] rounded-full"
               />
            </div>
         </motion.div>
       ))}
    </div>
  );
};

function Dashboard({ socket, onBack }) {
  const { user, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [active, setActive] = useState(false);
  const [image, setImage] = useState(null);
  const [lexiconMode, setLexiconMode] = useState('WORDS');
  const [detection, setDetection] = useState({ sign: "---", sentence: "", mode: "WORDS", fps: 0 });
  const [history, setHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [startFrame, setStartFrame] = useState(0);

  useEffect(() => {
    socket.on('prediction', (data) => {
      console.log('[Dashboard] Received prediction:', data);
      if (data.image) {
        setImage(`data:image/jpeg;base64,${data.image}`);
      }
      setDetection({
        sign: data.sign,
        sentence: data.sentence,
        mode: data.mode,
        fps: data.fps
      });
    });
    socket.on('stream_status', (data) => {
      console.log('[Dashboard] Stream status:', data);
    });
    socket.on('history_updated', (data) => {
      console.log('[Dashboard] History updated:', data);
      setHistory(data);
    });
    socket.emit('get_history');
    return () => {
      socket.off('prediction');
      socket.off('stream_status');
      socket.off('history_updated');
    };
  }, [socket]);

  const toggleCamera = () => {
    if (!active) socket.emit('start_detection');
    else socket.emit('stop_detection');
    setActive(!active);
  };

  const handleModeSwitch = (mode) => {
    setLexiconMode(mode);
    socket.emit('set_mode', mode);
  };

  const downloadHistory = () => {
    if (history.length === 0) return;
    const timestamp = new Date().toLocaleString();
    const content = `SignBridge AI - Session Archive\nDate: ${timestamp}\nUser: ${user?.name}\n-----------------------------\n\n` +
                    history.map((h, i) => `[${i + 1}] ${h.sentence}`).join('\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `SignBridge_Session_${new Date().getTime()}.txt`;
    link.click();
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('video/')) {
      alert('Please select a video file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('video', file);
    formData.append('startFrame', startFrame.toString());

    try {
      const API_BASE = import.meta.env.VITE_SERVER_URL || 'http://localhost:5000';
      const response = await axios.post(`${API_BASE}/api/upload-video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log('[Dashboard] Video upload successful:', response.data);
      // Auto-switch to Interpreter tab and activate the stream receiver
      setActive(true);
      setActiveTab('interpreter');
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed. Please check the network connection.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex bg-[var(--bg-primary)] min-h-screen font-inter text-[var(--text-primary)] transition-colors duration-300">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)} 
      />
      <div className="flex-1 lg:ml-64 flex flex-col min-w-0">
        <header className="h-20 md:h-24 px-6 md:px-12 flex items-center justify-between bg-[var(--nav-bg)] backdrop-blur-xl border-b border-[var(--border-color)] sticky top-0 z-50 transition-colors duration-300">
          <div className="flex items-center gap-4">
             <button 
               onClick={() => setSidebarOpen(true)}
               className="lg:hidden p-3 -ml-2 text-[#052E16] hover:bg-[#F0FDF4] rounded-2xl transition-colors active:scale-95 touch-none"
             >
                <LayoutGrid size={28} />
             </button>
              <Link to="/" className="hover:opacity-80 transition-opacity">
                 <div className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-white dark:bg-emerald-950/20 flex items-center justify-center border border-[var(--border-color)] shadow-sm overflow-hidden p-0.5">
                   <img src="/logo.png" className="w-full h-full object-contain logo-invert" alt="SignBridge" />
                 </div>
               </Link>
             <div className="hidden sm:block w-px h-6 bg-[#BBF7D0]/50 mx-2" />
             <div className="hidden sm:block w-2 h-2 bg-[#16A34A] rounded-full animate-pulse shadow-[0_0_12px_rgba(22,163,74,0.5)]" />
          </div>
          <div className="flex items-center gap-4 md:gap-8">
            <span className="hidden sm:inline text-[0.65rem] font-bold text-[#16A34A]/60 uppercase tracking-[0.3em]">Status: Verified</span>
            <div className="flex items-center gap-4 md:gap-8 pl-4 md:pl-8 sm:border-l border-[var(--border-color)]">
               <Bell className="text-[var(--text-secondary)] cursor-pointer" size={20} />
               <div onClick={() => setActiveTab('profile')} className="flex items-center gap-4 cursor-pointer group">
                  <div className="w-8 h-8 md:w-10 md:h-10 rounded-xl md:rounded-2xl bg-[#052E16] dark:bg-[#16A34A] flex items-center justify-center text-white text-xs font-black shadow-lg shadow-[#052E16]/20 transition-colors">
                    {user?.name?.charAt(0) || 'U'}
                  </div>
                  <span className="hidden md:inline text-xs font-black text-[var(--text-primary)] group-hover:text-[#16A34A] transition-colors uppercase tracking-widest">{user?.name}</span>
               </div>
               <button
                  onClick={toggleTheme}
                  className="p-2.5 text-[var(--text-secondary)] hover:text-[#16A34A] hover:bg-[var(--bg-tertiary)] rounded-2xl transition-all active:scale-95"
                  aria-label="Toggle Theme"
                >
                  {isDark ? <Sun size={20} /> : <Moon size={20} />}
                </button>
            </div>
          </div>
        </header>

        <main className="p-6 md:p-16 pb-48 max-w-[1700px] w-full mx-auto" role="main" aria-labelledby="dashboard-title">
          <AnimatePresence mode="wait">
            {activeTab === 'dashboard' ? (
              <motion.div key="stats" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-16 md:space-y-24">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-8">
                  <div>
                    <h1 id="dashboard-title" className="text-3xl md:text-5xl font-black tracking-tight mb-4 text-[var(--text-primary)]">Guardian Oversight</h1>
                    <p className="text-[#16A34A]/60 text-[0.65rem] md:text-[0.7rem] font-black uppercase tracking-[0.4em]">System Efficiency / Real-time Bio-metrics</p>
                  </div>
                  <div className="flex flex-wrap gap-4 md:gap-6 w-full md:w-auto">
                     <motion.button 
                       whileHover={{ scale: 1.05 }}
                       whileTap={{ scale: 0.95 }}
                       onClick={downloadHistory} 
                       className="flex-1 md:flex-none bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border-color)] px-6 md:px-8 py-3 md:py-4 rounded-2xl text-[0.7rem] font-black uppercase tracking-widest hover:bg-[var(--bg-tertiary)] transition-all shadow-sm"
                     >
                       Export Archive
                     </motion.button>
                     <motion.button 
                       whileHover={{ scale: 1.05 }}
                       whileTap={{ scale: 0.95 }}
                       onClick={() => setActiveTab('interpreter')} 
                       className="flex-1 md:flex-none bg-[#16A34A] text-white px-6 md:px-8 py-3 md:py-4 rounded-2xl text-[0.7rem] font-black uppercase tracking-widest hover:bg-[#15803d] transition-all shadow-xl shadow-[#16A34A]/20 flex items-center justify-center gap-3"
                     >
                       <Zap size={16} fill="white"/> Neural Interpreter
                     </motion.button>
                  </div>
                </div>
                
                <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] md:rounded-[3.5rem] p-8 md:p-16 shadow-[0_40px_100px_-30px_rgba(22,163,74,0.08)] relative overflow-hidden group transition-colors duration-300">
                   <div className="absolute top-0 right-0 w-64 h-64 bg-[#16A34A]/5 blur-[120px] rounded-full pointer-events-none" />
                   <div className="flex justify-between items-center mb-8 md:mb-12 relative z-10">
                      <h3 className="text-lg md:text-xl font-black text-[var(--text-primary)]">Growth Velocity <span className="text-[var(--text-tertiary)] text-xs font-medium block mt-1 tracking-normal">(Sign Acquisition Rates)</span></h3>
                      <div className="bg-[#16A34A]/10 px-4 md:px-6 py-2 md:py-3 rounded-xl md:rounded-2xl text-[8px] md:text-[10px] font-black uppercase tracking-[0.2em] text-[#16A34A] border border-[var(--border-color)]">Vitals: Optimal</div>
                   </div>
                   <div className="h-48 md:h-64 mb-8">
                     <WaveformChart />
                   </div>
                   <div className="flex flex-col md:flex-row justify-between mt-8 md:mt-12 pt-8 md:pt-12 border-t border-[var(--border-color)] relative z-10 gap-8">
                      <div className="flex gap-12 md:gap-20">
                         <div>
                            <p className="text-[0.6rem] md:text-[0.65rem] font-black text-[var(--text-tertiary)] uppercase tracking-[0.3em] mb-2">Total Vocabulary</p>
                            <h4 className="text-2xl md:text-3xl font-black text-[var(--text-primary)]">212 <span className="text-[#16A34A] text-[9px] md:text-[11px] ml-2 font-black">+24 UP</span></h4>
                         </div>
                         <div>
                            <p className="text-[0.6rem] md:text-[0.65rem] font-black text-[var(--text-tertiary)] uppercase tracking-[0.3em] mb-2">Model Confidence</p>
                            <h4 className="text-2xl md:text-3xl font-black text-[var(--text-primary)]">98.4%</h4>
                         </div>
                      </div>
                       <div className="flex items-center justify-center gap-3 text-[#16A34A] text-[9px] md:text-[10px] font-black tracking-[0.2em] uppercase bg-[#16A34A]/10 px-5 py-2.5 rounded-xl border border-[var(--border-color)]">
                          <Target size={18} /> Neural Interpreter 4.0
                       </div>
                    </div>
                 </div>
              </motion.div>
            ) : activeTab === 'profile' ? (
               <Profile key="profile" />
             ) : activeTab === 'categories' ? (
               <CategoriesView key="categories" />
            ) : activeTab === 'community' ? (
               <CommunityView key="community" />
            ) : activeTab === 'settings' ? (
               <SettingsView />
            ) : (
              <motion.div key="interpreter" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col lg:flex-row gap-8 md:gap-10 lg:h-[78vh]">
                  {/* LEFT: Functional Vision Stream */}
                  <div className="flex-[5] flex flex-col gap-6 md:gap-8">
                      <div className="aspect-video md:flex-1 bg-black border border-white/5 rounded-[2rem] md:rounded-[3rem] overflow-hidden relative group shadow-2xl">
                          {/* Floating Mode Controls */}
                          <div className="absolute top-6 right-6 z-30 flex bg-black/60 backdrop-blur-md p-1.5 rounded-2xl border border-white/10 shadow-2xl">
                             <button 
                               onClick={() => handleModeSwitch('ALPHABET')}
                               className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 ${lexiconMode === 'ALPHABET' ? 'bg-[#16A34A] text-white shadow-lg shadow-emerald-500/20' : 'text-white/40 hover:text-white hover:bg-white/5'}`}
                             >
                               Alphabet
                             </button>
                             <button 
                               onClick={() => handleModeSwitch('WORDS')}
                               className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all duration-300 ${lexiconMode === 'WORDS' ? 'bg-[#16A34A] text-white shadow-lg shadow-emerald-500/20' : 'text-white/40 hover:text-white hover:bg-white/5'}`}
                             >
                               Words
                             </button>
                          </div>
                          {active ? (
                            image ? (
                            <>
                               <img src={image} className="w-full h-full object-cover" alt="Feed" style={{ willChange: 'transform', transform: 'translateZ(0)' }} />
                               <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent pointer-events-none" />
                            </>
                            ) : (
                            <div className="w-full h-full bg-[#052E16] flex items-center justify-center">
                               <div className="text-center z-10 p-6">
                                  <div className="w-12 h-12 border-2 border-[#16A34A] border-t-transparent rounded-full animate-spin mb-4 mx-auto" />
                                  <p className="text-[#16A34A] text-xs font-black uppercase tracking-widest">Initializing Core...</p>
                               </div>
                            </div>
                            )
                         ) : (
                             <div className="w-full h-full bg-[#052E16] flex items-center justify-center relative">
                               <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_center,_white_1px,_transparent_1px)] bg-[size:40px_40px]" />
                               <div className="text-center z-10 p-6">
                                  <div className="w-16 h-16 md:w-24 md:h-24 bg-[#16A34A] text-white rounded-full flex items-center justify-center mx-auto mb-6 md:mb-8 border border-[#16A34A]/20 shadow-[0_0_50px_rgba(22,163,74,0.3)] animate-pulse">
                                     <Video size={32} md:size={48} strokeWidth={1} />
                                  </div>
                                  <h3 className="text-lg md:text-2xl font-black text-white tracking-[0.15em] md:tracking-[0.2em] uppercase">Vision Core Standby</h3>
                                  <p className="text-[#16A34A] text-[9px] md:text-[10px] font-black uppercase tracking-[0.4em] mt-3 md:mt-4">Initialize terminal to begin uplink</p>
                               </div>
                             </div>
                          )}
                       </div>
                        

                          <div className="absolute bottom-6 md:bottom-12 left-1/2 -translate-x-1/2 z-20 flex gap-4 w-max">
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={toggleCamera}
                              aria-label={active ? 'Stop real-time ASL detection' : 'Start real-time ASL detection using webcam'}
                              className={`flex-1 md:flex-none px-6 md:px-12 py-5 rounded-[1.5rem] md:rounded-[2rem] font-black uppercase tracking-[0.2em] md:tracking-[0.3em] text-[11px] md:text-[12px] flex items-center justify-center gap-4 transition-all ${active ? 'bg-[#EF4444]/20 backdrop-blur-xl text-white border border-red-500/50 shadow-2xl' : 'bg-[var(--bg-primary)] text-[var(--text-primary)] shadow-2xl hover:bg-[var(--bg-tertiary)] border border-[var(--border-color)]'}`}
                            >
                              {active ? <><VideoOff size={18}/> Stop</> : <><Power size={18}/> Start Core</>}
                            </motion.button>
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => document.getElementById('video-upload').click()}
                              disabled={uploading}
                              aria-label={uploading ? 'Uploading video file' : 'Upload video file for ASL detection'}
                              className={`flex-1 md:flex-none px-8 md:px-10 py-4 md:py-5 rounded-2xl md:rounded-[2rem] font-black uppercase tracking-[0.2em] md:tracking-[0.3em] text-[10px] md:text-[12px] flex items-center justify-center gap-3 transition-all bg-[#16A34A] text-white shadow-2xl hover:bg-[#15803d] disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                              <Upload size={16} /> {uploading ? 'Uploading...' : 'Upload Video'}
                            </motion.button>
                            <input
                              id="video-upload"
                              type="file"
                              accept="video/*"
                              onChange={handleVideoUpload}
                              aria-describedby="video-upload-desc"
                              style={{ display: 'none' }}
                            />
                            <div id="video-upload-desc" style={{ display: 'none' }}>Select a video file to upload for American Sign Language detection</div>
                            <input
                              type="number"
                              min="0"
                              value={startFrame}
                              onChange={(e) => setStartFrame(Math.max(0, parseInt(e.target.value) || 0))}
                              placeholder="Start frame"
                              className="w-20 md:w-24 px-3 py-4 md:py-5 rounded-2xl md:rounded-[2rem] bg-[var(--bg-secondary)] border border-[var(--border-color)] text-white text-xs md:text-sm font-black uppercase placeholder:text-white/30"
                              aria-label="Start frame number"
                            />
                         </div>
                     </div>


                  {/* RIGHT / BOTTOM: High-End Neural HUD */}
                  <div className="flex-[2] flex flex-col gap-8 md:gap-10">
                     {/* Transcription Terminal Panel */}
                     <div className="bg-[#052E16] border border-white/5 rounded-[2rem] md:rounded-[3rem] p-8 md:p-10 flex flex-col shadow-2xl relative overflow-hidden min-h-[300px] md:h-1/2 group">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-[#16A34A]/10 blur-[100px] rounded-full pointer-events-none" />
                        
                        <div className="flex justify-between items-center mb-8 md:mb-10 relative z-10">
                           <div className="flex items-center gap-4">
                              <Terminal size={18} className="text-[#16A34A]" />
                              <span className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.4em] text-white/40">Transcription Terminal</span>
                           </div>
                           <div className="flex gap-2">
                              <div className="w-1.5 h-1.5 bg-[#4ADE80] rounded-full animate-bounce" />
                              <div className="w-1.5 h-1.5 bg-[#4ADE80] rounded-full animate-bounce [animation-delay:0.2s]" />
                              <div className="w-1.5 h-1.5 bg-[#4ADE80] rounded-full animate-bounce [animation-delay:0.4s]" />
                           </div>
                        </div>

                        <div className="flex-1 overflow-y-auto scroll-none pr-4 min-h-[150px]">
                              <div className="h-full flex flex-col justify-end space-y-6">
                                 <div className="space-y-1">
                                    <span className="text-[8px] md:text-[9px] font-black text-[#16A34A] uppercase tracking-[0.4em]">Current Sign::</span>
                                    <h4 className="text-xl md:text-2xl font-black text-[#16A34A] tracking-tighter italic uppercase truncate">{detection.sign || "..."}</h4>
                                 </div>
                                 <div className="space-y-2">
                                    <span className="text-[8px] md:text-[9px] font-black text-white/30 uppercase tracking-[0.4em]">Output Stream::</span>
                                    <h2 className="text-lg md:text-xl font-black tracking-tight text-white leading-relaxed italic uppercase break-words px-1">
                                       {detection.sentence || "..."}
                                    </h2>
                                 </div>
                              </div>
                        </div>

                        <button 
                          onClick={() => socket.emit('clear_sentence')}
                          className="absolute bottom-6 md:bottom-10 right-6 md:right-10 p-3 md:p-4 bg-white/5 hover:bg-white/10 text-white/30 hover:text-white rounded-xl md:rounded-2xl border border-white/5 transition-all"
                        >
                          <RefreshCcw size={18} />
                        </button>
                     </div>

                     {/* Session History Strip */}
                     <div className="flex-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] md:rounded-[3.5rem] p-8 md:p-12 flex flex-col shadow-sm overflow-hidden group min-h-[300px] transition-colors duration-300">
                        <div className="flex justify-between items-center mb-8 md:mb-10">
                           <h4 className="text-[9px] md:text-[10px] font-black uppercase tracking-[0.3em] text-[var(--text-primary)] flex items-center gap-4">
                              <HistoryIcon size={18} className="text-[#16A34A]" /> Session Archive
                           </h4>
                           <div className="flex gap-3 md:gap-4">
                              <button onClick={downloadHistory} className="p-2.5 md:p-3 bg-[#16A34A]/10 text-[#16A34A] rounded-lg md:rounded-xl border border-[var(--border-color)] hover:bg-[#16A34A] hover:text-white transition-all">
                                 <Download size={18} />
                              </button>
                              <button onClick={() => socket.emit('clear_history')} className="p-2.5 md:p-3 bg-red-500/10 text-red-500 rounded-lg md:rounded-xl border border-red-500/20 hover:bg-red-500 hover:text-white transition-all">
                                 <RefreshCcw size={18} />
                              </button>
                           </div>
                        </div>

                        <div className="flex-1 overflow-auto scroll-none space-y-6">
                           {history.length === 0 ? (
                              <div className="h-full flex flex-col items-center justify-center text-[var(--text-tertiary)] py-10 opacity-30">
                                 <Activity size={48} strokeWidth={1} />
                                 <p className="mt-4 text-[9px] md:text-[10px] font-black uppercase tracking-[0.4em]">Listening...</p>
                              </div>
                           ) : (
                              history.slice().reverse().map((h, i) => (
                                 <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ ease: "circOut", duration: 0.09 }} key={h.id || i} className="flex items-center gap-5 group/hist border-b border-[var(--border-color)] pb-4 last:border-0">
                                    <div className="w-1.5 h-1.5 bg-[var(--border-color)] group-hover/hist:bg-[#16A34A] transition-colors" />
                                    <div className="flex-1">
                                       <p className="text-xs md:text-sm font-black text-[var(--text-primary)] uppercase italic leading-tight mb-1">{h.sentence}</p>
                                       <span className="text-[8px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest">{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                    </div>
                                 </motion.div>
                              ))
                           )}
                         </div>
                      </div>
                  </div>
               </motion.div>
             )}
           </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

const Leaf = ({ size, className, fill = "none" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 9.1-1.1 6.3-5 8.9-9 8.9Z" />
    <path d="M7 20c0-3 2-6 9-9" />
  </svg>
);

const CheckCircle = ({ size, className }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

const ConstructionView = ({ title, description, icon: Icon }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="flex flex-col items-center justify-center h-[70vh] text-center space-y-8">
     <div className="w-24 h-24 bg-emerald-50 rounded-[2rem] flex items-center justify-center text-emerald-500 shadow-xl shadow-emerald-500/10 border border-emerald-100">
        <Icon size={48} />
     </div>
     <div>
       <h1 className="text-4xl font-black text-[#052E16] tracking-tight">{title}</h1>
       <p className="text-gray-400 font-medium max-w-md mx-auto mt-4 leading-relaxed">{description}</p>
     </div>
     <div className="px-6 py-3 bg-white border border-gray-100 rounded-full shadow-sm flex items-center gap-3 text-xs font-black uppercase tracking-widest text-[#16A34A]">
       <Sparkles size={16} /> Module Unlocking Soon
     </div>
  </motion.div>
);

export default Dashboard;
