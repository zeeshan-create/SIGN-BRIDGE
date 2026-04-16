import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from './AuthContext';
import { useTheme } from './ThemeContext';
import { Settings, Save, AlertCircle, ShieldCheck, Zap, Eye, Sliders, Database, Smartphone, CheckCircle, Layers, Sun, Moon } from 'lucide-react';

export default function SettingsView() {
  const { user } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  
  // States
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    password: '••••••••',
  });

  const [toggles, setToggles] = useState({
    ultraLowLatency: true,
    hapticFeedback: true,
    reducedMotion: false,
    hardwareAcceleration: true,
    autoConnectData: false
  });

  const [installPrompt, setInstallPrompt] = useState(null);
  const [isStandalone, setIsStandalone] = useState(false);

  React.useEffect(() => {
    // Check if already in standalone mode
    if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
      setIsStandalone(true);
    }

    const handler = (e) => {
      e.preventDefault();
      setInstallPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === 'accepted') {
      setInstallPrompt(null);
    }
  };

  const [confidence, setConfidence] = useState(85);

  const handleToggle = (key) => {
    setToggles(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    }, 1500);
  };

  return (
    <motion.div key="settings" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="pb-32 transition-colors duration-300">
      <div className="flex justify-between items-end mb-12">
        <div>
          <h1 className="text-4xl md:text-5xl font-black tracking-tight mb-4 text-[var(--text-primary)]">System Preferences</h1>
          <p className="text-[#16A34A]/60 text-[0.7rem] font-black uppercase tracking-[0.4em]">Engine Calibration & Tolerances</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={toggleTheme}
            className="p-4 rounded-2xl bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-all active:scale-95 shadow-lg"
            aria-label="Toggle Theme"
          >
            {isDark ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button 
            onClick={handleSave}
            disabled={isSaving}
            className={`px-8 py-4 rounded-[1.5rem] text-[0.75rem] font-black uppercase tracking-widest transition-all shadow-xl flex items-center gap-3 ${
              saveSuccess 
                ? 'bg-[#052E16] text-[#BBF7D0] shadow-[#052E16]/20'
                : 'bg-[#16A34A] text-white hover:bg-[#15803d] shadow-[#16A34A]/20'
            }`}
          >
            {isSaving ? (
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 0.3, ease: 'linear' }}>
                <Settings size={16} />
              </motion.div>
            ) : saveSuccess ? (
              <><ShieldCheck size={16} /> Saved</>
            ) : (
              <><Save size={16} /> Apply Config</>
            )}
          </button>
        </div>
      </div>

      <div className="grid xl:grid-cols-2 gap-10">
        
        {/* LEFT COLUMN */}
        <div className="space-y-10">
          
          {/* Neural Engine Panel */}
          <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-10 shadow-sm relative overflow-hidden">
             <div className="absolute top-0 right-0 p-8 text-[#16A34A]/10 pointer-events-none"><Zap size={100} /></div>
             <h3 className="text-xl font-black text-[var(--text-primary)] mb-8 flex items-center gap-3 relative z-10"><Sliders size={20} className="text-[#16A34A]"/> Neural Calibration</h3>
             
             <div className="space-y-8 relative z-10">
                <div>
                   <div className="flex justify-between mb-4">
                     <span className="text-xs font-bold text-[var(--text-tertiary)] uppercase tracking-widest">Confidence Threshold</span>
                     <span className="text-xs font-black text-[#16A34A]">{confidence}%</span>
                   </div>
                   <input 
                     type="range" min="50" max="99" value={confidence} onChange={(e) => setConfidence(e.target.value)}
                     className="w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer accent-[#16A34A]"
                   />
                   <p className="text-[10px] text-[var(--text-tertiary)] font-medium mt-2">Higher thresholds reduce false positives but may drop subtle signs.</p>
                </div>

                <div className="flex items-center justify-between py-4 border-t border-[var(--border-color)]">
                   <div>
                      <p className="text-sm font-bold text-[var(--text-primary)]">Ultra-Low Latency Mode</p>
                      <p className="text-[10px] text-[var(--text-tertiary)] tracking-wider">Bypass secondary validation for raw speed.</p>
                   </div>
                   <button onClick={() => handleToggle('ultraLowLatency')} className={`w-12 h-6 rounded-full transition-colors relative ${toggles.ultraLowLatency ? 'bg-[#16A34A]' : 'bg-[var(--bg-tertiary)]'}`}>
                      <motion.div className="w-4 h-4 bg-white rounded-full absolute top-1" animate={{ left: toggles.ultraLowLatency ? '26px' : '4px' }} transition={{ type: 'spring', stiffness: 500, damping: 30 }} />
                   </button>
                </div>

                <div className="flex items-center justify-between py-4 border-t border-[var(--border-color)]">
                   <div>
                      <p className="text-sm font-bold text-[var(--text-primary)]">Hardware Acceleration</p>
                      <p className="text-[10px] text-[var(--text-tertiary)] tracking-wider">Utilize local WebGL/CUDA processors.</p>
                   </div>
                   <button onClick={() => handleToggle('hardwareAcceleration')} className={`w-12 h-6 rounded-full transition-colors relative ${toggles.hardwareAcceleration ? 'bg-[#16A34A]' : 'bg-[var(--bg-tertiary)]'}`}>
                      <motion.div className="w-4 h-4 bg-white rounded-full absolute top-1" animate={{ left: toggles.hardwareAcceleration ? '26px' : '4px' }} transition={{ type: 'spring', stiffness: 500, damping: 30 }} />
                   </button>
                </div>
             </div>
          </div>

          {/* User Parameters */}
          <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-10 shadow-sm">
             <h3 className="text-xl font-black text-[var(--text-primary)] mb-8 flex items-center gap-3"><Database size={20} className="text-[#16A34A]"/> Account Parameters</h3>
             
             <div className="space-y-6">
                <div>
                   <label className="text-[10px] font-black text-[var(--text-tertiary)] uppercase tracking-widest ml-4">Full Identity</label>
                   <input type="text" value={formData.name} onChange={(e) => setFormData({...formData, name: e.target.value})} className="w-full mt-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-2xl px-6 py-4 text-sm font-bold text-[var(--text-primary)] focus:outline-none focus:border-[#BBF7D0] focus:ring-4 focus:ring-[#BBF7D0]/20 transition-all" />
                </div>
                <div>
                   <label className="text-[10px] font-black text-[var(--text-tertiary)] uppercase tracking-widest ml-4">Security Key (Email)</label>
                   <input type="email" value={formData.email} onChange={(e) => setFormData({...formData, email: e.target.value})} className="w-full mt-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-2xl px-6 py-4 text-sm font-bold text-[var(--text-primary)] focus:outline-none focus:border-[#BBF7D0] focus:ring-4 focus:ring-[#BBF7D0]/20 transition-all" />
                </div>
                <div>
                   <label className="text-[10px] font-black text-[var(--text-tertiary)] uppercase tracking-widest ml-4">Passcode</label>
                   <input type="password" value={formData.password} onChange={(e) => setFormData({...formData, password: e.target.value})} className="w-full mt-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-2xl px-6 py-4 text-sm font-bold text-[var(--text-primary)] focus:outline-none focus:border-[#BBF7D0] focus:ring-4 focus:ring-[#BBF7D0]/20 transition-all" />
                </div>
             </div>
          </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="space-y-10">

           {/* UI Tolerances */}
           <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-10 shadow-sm relative overflow-hidden">
             <h3 className="text-xl font-black text-[var(--text-primary)] mb-8 flex items-center gap-3"><Eye size={20} className="text-[#16A34A]"/> Interface Tolerances</h3>
             
             <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-[var(--bg-tertiary)] rounded-2xl">
                   <div>
                      <p className="text-sm font-bold text-[var(--text-primary)]">Reduced Motion</p>
                      <p className="text-[10px] text-[var(--text-tertiary)] mt-1">Strip non-essential UI animations for performance.</p>
                   </div>
                   <button onClick={() => handleToggle('reducedMotion')} className={`w-10 h-5 rounded-full transition-colors relative ${toggles.reducedMotion ? 'bg-[#16A34A]' : 'bg-[var(--bg-tertiary)]'}`}>
                      <motion.div className="w-3 h-3 bg-white rounded-full absolute top-1" animate={{ left: toggles.reducedMotion ? '22px' : '4px' }} transition={{ type: 'spring', stiffness: 500 }} />
                   </button>
                </div>

                <div className="flex items-center justify-between p-4 bg-[var(--bg-tertiary)] rounded-2xl">
                   <div>
                      <p className="text-sm font-bold text-[var(--text-primary)]">Haptic Visual Feedback</p>
                      <p className="text-[10px] text-[var(--text-tertiary)] mt-1">Render micro-vibrations and button press depths.</p>
                   </div>
                   <button onClick={() => handleToggle('hapticFeedback')} className={`w-10 h-5 rounded-full transition-colors relative ${toggles.hapticFeedback ? 'bg-[#16A34A]' : 'bg-[var(--bg-tertiary)]'}`}>
                      <motion.div className="w-3 h-3 bg-white rounded-full absolute top-1" animate={{ left: toggles.hapticFeedback ? '22px' : '4px' }} transition={{ type: 'spring', stiffness: 500 }} />
                   </button>
                </div>
             </div>
          </div>

          {/* Privacy & Sync */}
          <div className="bg-[#052E16] border border-emerald-900 rounded-[2.5rem] p-10 shadow-2xl relative overflow-hidden">
             <div className="absolute -top-10 -right-10 w-40 h-40 bg-emerald-500 rounded-full blur-[80px] opacity-20 pointer-events-none" />
             <h3 className="text-xl font-black text-white mb-8 flex items-center gap-3 relative z-10"><Smartphone size={20} className="text-[#16A34A]"/> Privacy & Data Link</h3>
             
             <div className="space-y-6 relative z-10">
                <div className="flex items-center justify-between pb-6 border-b border-emerald-900/50">
                   <div>
                      <p className="text-sm font-bold text-white">Cloud Translation Sync</p>
                      <p className="text-[10px] text-emerald-100/50 mt-1">Anonymously send difficult signs to cloud for engine training.</p>
                   </div>
                   <button onClick={() => handleToggle('autoConnectData')} className={`w-12 h-6 rounded-full transition-colors relative ${toggles.autoConnectData ? 'bg-[#16A34A]' : 'bg-white/10'}`}>
                      <motion.div className="w-4 h-4 bg-white rounded-full absolute top-1" animate={{ left: toggles.autoConnectData ? '26px' : '4px' }} transition={{ type: 'spring', stiffness: 500 }} />
                   </button>
                </div>

                <div className="bg-emerald-900/40 border border-emerald-800 rounded-2xl p-6 flex flex-col gap-4">
                   <div className="flex items-start gap-4">
                      <AlertCircle className="text-amber-400 flex-shrink-0 mt-1" size={18} />
                      <div>
                         <p className="text-sm font-bold text-white mb-1">Local Processing Guarantee</p>
                         <p className="text-xs text-emerald-100/60 leading-relaxed">SignBridge prioritizes privacy. Your camera feed is processed entirely locally on this device's GPU. No imagery ever leaves your workstation.</p>
                      </div>
                   </div>
                </div>
             </div>
          </div>

          {/* Hybrid Device Status */}
          <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[2.5rem] p-10 shadow-sm relative overflow-hidden">
             <div className="absolute top-0 right-0 p-8 text-[#16A34A]/5 pointer-events-none self-end"><Smartphone size={80} /></div>
             <h3 className="text-xl font-black text-[var(--text-primary)] mb-8 flex items-center gap-3 relative z-10"><Layers size={20} className="text-[#16A34A]"/> Hybrid Device Link</h3>
             
             <div className="space-y-6 relative z-10">
                <div className="flex items-center justify-between p-6 bg-[var(--bg-tertiary)] rounded-2xl border border-[var(--border-color)]">
                   <div>
                      <p className="text-sm font-bold text-[var(--text-primary)]">{isStandalone ? 'App Installed' : 'Browser Mode'}</p>
                      <p className="text-[10px] text-[var(--text-tertiary)] mt-1 uppercase tracking-widest font-black">{isStandalone ? 'Running as standalone system' : 'Running via web interface'}</p>
                   </div>
                   {isStandalone ? (
                      <div className="w-10 h-10 rounded-full bg-[#16A34A] text-white flex items-center justify-center">
                         <CheckCircle size={20} />
                      </div>
                   ) : (
                      <button 
                        disabled={!installPrompt}
                        onClick={handleInstall}
                        className={`px-6 py-2.5 rounded-xl text-[9px] font-black uppercase tracking-widest transition-all ${installPrompt ? 'bg-[#16A34A] text-white hover:bg-[#15803d]' : 'bg-[var(--bg-secondary)] text-[var(--text-tertiary)] cursor-not-allowed'}`}
                      >
                        {installPrompt ? 'Launch Install' : 'Ready for App'}
                      </button>
                   )}
                </div>
                <p className="text-[9px] text-[var(--text-tertiary)] leading-relaxed font-medium">
                  SignBridge is built as a Progressive Web App (PWA). You can install it on your mobile homescreen or desktop taskbar for a native experience without visiting the URL.
                </p>
             </div>
          </div>

        </div>
      </div>
    </motion.div>
  );
}
