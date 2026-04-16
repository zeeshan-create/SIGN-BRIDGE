import React from 'react';
import { Play, Languages, Globe2, Link2, Zap, Fingerprint, Activity, Shield } from 'lucide-react';
import { motion } from 'framer-motion';

function Landing({ onStart }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-emerald-50/30 to-white text-[#0f172a] font-inter overflow-hidden">
      
      {/* Background subtle grid pattern */}
      <div className="fixed inset-0 opacity-30 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(#10b981_1px,transparent_1px)] bg-[size:32px_32px]" />
      </div>
      {/* Navbar */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
        className="max-w-[1400px] mx-auto px-10 h-20 flex justify-between items-center bg-white/90 backdrop-blur-md sticky top-0 z-50 rounded-b-3xl mt-0 will-change-transform transform translate-z-0 contain layout paint style"
      >
        <div className="flex items-center gap-3 font-bold text-xl tracking-tight">
          <div className="w-10 h-10 bg-[#10b981] rounded-xl flex items-center justify-center text-white shadow-lg shadow-emerald-200">
            <Activity size={20} />
          </div>
          <span className="font-black text-xl">SignBridge</span>
        </div>
        
        <div className="flex items-center gap-1 text-sm font-medium text-[#475569]">
          {['Features', 'Converter', 'Dashboard', 'Community'].map((item, i) => (
            <motion.a 
              key={item} 
              href="#" 
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.0 + (i * 0.016) }}
              whileHover={{ y: -2, color: "#10b981" }}
              className="px-4 py-2 rounded-xl hover:bg-emerald-50 transition-colors duration-200"
            >
              {item}
            </motion.a>
          ))}
        </div>
        
        <div className="flex items-center gap-4">
          <motion.a 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.08 }}
            href="/login" 
            className="text-[#475569] font-medium px-4 py-2 rounded-xl hover:bg-gray-50 transition-colors duration-200"
          >
            Login
          </motion.a>
          <motion.button 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.09 }}
            onClick={onStart}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="bg-[#10b981] text-white px-6 py-2.5 rounded-xl font-bold shadow-lg shadow-emerald-200 hover:bg-emerald-600 transition-colors transition-transform duration-200 gpu-accelerate"
          >
            Get Started Free
          </motion.button>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <main className="max-w-[1400px] mx-auto px-10 pt-32 pb-20 relative z-10">
        <div className="text-center mb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="inline-flex items-center gap-2 bg-emerald-50 text-emerald-700 px-5 py-2 rounded-full text-xs font-black uppercase tracking-widest mb-10 border border-emerald-100 shadow-sm"
            style={{ willChange: 'transform, opacity' }}
          >
            <Zap size={14} fill="currentColor" />
            Real-time AI Sign Language
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12, duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="text-6xl md:text-7xl lg:text-8xl font-black leading-[1.05] tracking-[-0.04em] mb-8 text-slate-900"
            style={{ willChange: 'transform, opacity' }}
          >
            Break barriers.<br />
            <span className="text-emerald-500">Understand every sign.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="text-xl text-slate-500 max-w-2xl mx-auto mb-16 leading-relaxed"
            style={{ willChange: 'transform, opacity' }}
          >
            The world's most accurate real-time ASL interpretation platform. Translate sign language to text instantly with our neural vision engine.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18, duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
            className="flex items-center justify-center gap-4 mb-24"
            style={{ willChange: 'transform, opacity' }}
          >
            <motion.button 
              onClick={onStart}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              className="group bg-[#10b981] text-white px-10 py-5 rounded-2xl text-base font-bold shadow-2xl shadow-emerald-200 hover:bg-emerald-600 transition-transform flex items-center gap-3 gpu-accelerate"
            >
              Start Live Interpreter
              <div className="bg-white/20 p-1.5 rounded-xl group-hover:bg-white/30 transition-colors">
                <Play size={16} fill="currentColor" />
              </div>
            </motion.button>
            
          <motion.button 
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="bg-white text-slate-700 px-8 py-5 rounded-2xl text-base font-bold border border-slate-200 shadow-sm hover:border-emerald-300 transition-transform transition-colors gpu-accelerate"
          >
              Try Converter
            </motion.button>
          </motion.div>

         {/* Feature Cards */}
         <div className="grid grid-cols-1 md:grid-cols-3 gap-10 mt-16">
           {[
             { 
               icon: Zap, 
               title: "Zero Latency",
               desc: "Real-time translation optimized for blink-of-an-eye speed using edge neural processing.",
               color: "bg-[#10b981]",
               accent: "#10b981"
             },
             { 
               icon: Shield, 
               title: "Privacy First",
               desc: "Local-only interpretation. Your vision feed never leaves your personal device architecture.",
               color: "bg-[#022c22]",
               accent: "#022c22"
             },
             { 
               icon: Activity, 
               title: "Standard ASL",
               desc: "Comprehensive support for the American Sign Language lexicon, from ABCs to complex phrases.",
               color: "bg-[#d1fae5]",
               accent: "#059669"
             }
           ].map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 60 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.04 + (i * 0.06), duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
              whileHover={{ y: -12 }}
              className="bg-white border border-white rounded-[3rem] p-10 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.03)] transition-transform duration-300 group relative overflow-hidden gpu-accelerate"
              style={{ willChange: 'transform' }}
            >
               <motion.div 
                 initial={{ scale: 0.8, opacity: 0 }}
                 animate={{ scale: 1, opacity: 1 }}
                 transition={{ delay: 0.05 + (i * 0.024), type: "spring" }}
                 className={`w-20 h-20 ${feature.color} rounded-[2rem] flex items-center justify-center text-white mb-8 shadow-xl group-hover:scale-110 transition-transform duration-300`}
                 style={{ boxShadow: `0 20px 40px -10px ${feature.accent}40` }}
               >
                 <feature.icon size={32} strokeWidth={2} />
               </motion.div>
               
               <motion.h3 
                 className="text-3xl font-black mb-6 text-[#022c22] tracking-tight"
                 initial={{ x: -10, opacity: 0 }}
                 animate={{ x: 0, opacity: 1 }}
                 transition={{ delay: 0.06 + (i * 0.024) }}
               >
                 {feature.title}
               </motion.h3>
               
                <motion.div 
                  className="w-1 h-24 rounded-full mb-6 origin-top"
                  style={{ backgroundColor: feature.accent, opacity: 0.2 }}
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  transition={{ delay: 0.06 + (i * 0.024), duration: 0.24 }}
                />
               
               <motion.p 
                 className="text-lg text-gray-500 leading-relaxed font-medium"
                 initial={{ opacity: 0 }}
                 animate={{ opacity: 1 }}
                 transition={{ delay: 0.06 + (i * 0.024) }}
               >
                 {feature.desc}
               </motion.p>
             </motion.div>
           ))}
         </div>
      </main>
    </div>
  );
}

export default Landing;
