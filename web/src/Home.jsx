import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Check, ArrowRight, Zap, ShieldCheck, Library, Sparkles, 
  Globe, Cpu, Database, Activity, Code2, Network, Radio, Layers, Eye
} from 'lucide-react';
import { Link } from 'react-router-dom';

const MotionLink = motion(Link);

const Leaf = ({ size, className, fill = "none" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 1 9.1-1.1 6.3-5 8.9-9 8.9Z" />
    <path d="M7 20c0-3 2-6 9-9" />
  </svg>
);

// --- Vertical Roaming Tech Field Component ---
const NeuralTechField = () => {
  const column1 = [
    { name: 'MediaPipe Hands', icon: Activity },
    { name: 'LSTM Core', icon: Cpu },
    { name: 'CUDA Acceleration', icon: Zap },
    { name: 'Socket.IO Bridge', icon: Radio }
  ];
  const column2 = [
    { name: 'Dense Neural Net', icon: Layers },
    { name: 'ReLU Activation', icon: Sparkles },
    { name: 'Label Encoder', icon: Database },
    { name: 'OpenCV Engine', icon: Eye }
  ];
  const column3 = [
    { name: 'Dropout (0.2)', icon: ShieldCheck },
    { name: 'Softmax Classifier', icon: Network },
    { name: 'Edge Processing', icon: Zap },
    { name: 'Neural Sync', icon: Radio }
  ];

  const Column = ({ items, speed, reverse = false }) => (
    <motion.div 
      initial={{ y: reverse ? '-50%' : '0%' }}
      animate={{ y: reverse ? '0%' : '-50%' }}
      transition={{ duration: speed, repeat: Infinity, ease: "linear" }}
      className="flex flex-col gap-10 py-10"
    >
      {[...items, ...items].map((item, i) => (
        <div 
          key={i} 
          className="bg-[var(--bg-secondary)] border border-[var(--border-color)]/50 p-10 rounded-[3rem] shadow-[0_30px_60px_-15px_rgba(22,163,74,0.06)] flex flex-col items-center justify-center gap-6 group hover:border-[#16A34A] transition-all hover:-translate-y-3 cursor-pointer w-56"
        >
          <div className="w-16 h-16 bg-[var(--bg-tertiary)] text-[#16A34A] rounded-2xl flex items-center justify-center group-hover:bg-[#16A34A] group-hover:text-white transition-all shadow-inner">
            <item.icon size={28} />
          </div>
          <span className="text-[11px] font-black uppercase tracking-[0.3em] text-[var(--text-primary)] text-center leading-tight">{item.name}</span>
        </div>
      ))}
    </motion.div>
  );

  return (
    <div className="relative h-[900px] w-full flex items-center justify-center overflow-hidden bg-[var(--bg-secondary)] rounded-[5rem] my-32 mx-auto max-w-[1400px] border border-[var(--border-color)] shadow-inner">
      {/* Background Text */}
      <div className="absolute inset-0 flex items-center justify-center select-none pointer-events-none opacity-[0.03]">
         <div className="text-[35rem] font-black tracking-tighter text-[#052E16]">CORE</div>
      </div>

      <div className="relative z-10 flex gap-16 rotate-[-4deg] scale-110">
        <Column items={column1} speed={25} />
        <Column items={column2} speed={35} reverse={true} />
        <Column items={column3} speed={20} />
      </div>

      {/* Gradients to fade top/bottom */}
      <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-[#F0FDF4] via-transparent to-[#F0FDF4] z-20" />
    </div>
  );
};

function Home() {
  const features = [
    { 
      title: 'Zero Latency', 
      desc: 'Real-time translation optimized for blink-of-an-eye speed using edge neural processing.',
      icon: Zap,
      color: 'bg-[#16A34A]',
      shadow: 'shadow-[#16A34A]/20'
    },
    { 
      title: 'Privacy First', 
      desc: 'Local-only interpretation. Your vision feed never leaves your personal device architecture.',
      icon: ShieldCheck,
      color: 'bg-[#052E16]',
      shadow: 'shadow-[#052E16]/20'
    },
    { 
      title: 'Standard ASL', 
      desc: 'Comprehensive support for the American Sign Language lexicon, from ABCs to complex phrases.',
      icon: Library,
      color: 'bg-[#BBF7D0]',
      textColor: 'text-[#052E16]',
      shadow: 'shadow-[#BBF7D0]/20'
    },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[var(--bg-primary)] font-inter selection:bg-[#BBF7D0] selection:text-[#052E16] transition-colors duration-300"
    >
      {/* Editorial Hero Section */}
      <section className="relative max-w-[1600px] mx-auto px-6 md:px-10 py-24 md:py-64 text-center overflow-hidden">
        <div className="absolute top-10 md:top-20 left-1/2 -translate-x-1/2 w-[300px] md:w-[700px] h-[300px] md:h-[350px] bg-[#F0FDF4] blur-[60px] md:blur-[80px] rounded-full pointer-events-none -z-10 opacity-70" />
        
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-3 bg-[#F0FDF4] text-[#16A34A] px-6 py-2.5 rounded-full text-[0.65rem] md:text-[0.7rem] font-black uppercase tracking-[0.3em] mb-12 md:mb-16 border border-[#BBF7D0] shadow-sm"
        >
          <Leaf size={14} fill="currentColor" />
          Neural Interpreter 4.0 Live
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.02, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="text-4xl sm:text-7xl md:text-[8rem] font-black tracking-tighter text-[var(--text-primary)] leading-[1] md:leading-[0.9] mb-8 md:mb-16"
        >
          SignBridge.<br/>
          <span className="text-[#16A34A] relative inline-block">
            Where Signs Become Voice.
            <svg className="absolute -bottom-6 md:-bottom-10 left-0 w-full h-4 md:h-6" viewBox="0 0 100 10" preserveAspectRatio="none">
              <path 
                d="M2,5 Q50,10 98,5" 
                stroke="#10B981" 
                strokeWidth="5" 
                strokeLinecap="round" 
                fill="none" 
                className="opacity-20"
              />
            </svg>
          </span>
        </motion.h1>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.04 }}
          className="max-w-[1000px] mx-auto mb-16 md:mb-24 space-y-8 md:space-y-12"
        >
          <p className="text-xl md:text-4xl text-[var(--text-primary)] leading-relaxed font-bold tracking-tight">
            The most <span className="text-[#10b981] relative inline-block px-1">intuitive</span> ASL interpretation <span className="bg-emerald-500/10 px-3 py-1 rounded-xl">engine</span> ever built.
          </p>
          
          <div className="bg-[var(--nav-bg)] backdrop-blur-3xl border border-[var(--border-color)] p-4 md:p-10 rounded-[1.5rem] md:rounded-[2.5rem] flex flex-wrap justify-center items-center gap-x-6 md:gap-x-12 gap-y-3 md:gap-y-6 text-[0.6rem] md:text-[0.75rem] font-black uppercase tracking-[0.2em] md:tracking-[0.45em] text-[var(--text-tertiary)] shadow-sm">
            <span className="hover:text-[#16A34A] transition-colors duration-500 cursor-default">SignBridge</span>
            <div className="hidden sm:block w-1.5 h-1.5 rounded-full bg-[#16A34A]/20" />
            <span className="hover:text-[#16A34A] transition-colors duration-500 cursor-default underline decoration-[#16A34A]/20 underline-offset-8">Understanding. One Sign at a Time.</span>
            <div className="hidden sm:block w-1.5 h-1.5 rounded-full bg-[#16A34A]/20" />
            <span className="hover:text-[#16A34A] transition-colors duration-500 cursor-default text-emerald-600/50">ASL → Text. Instantly.</span>
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.06 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-6 md:gap-8"
        >
          <MotionLink
            to="/dashboard"
            whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(16, 185, 129, 0.5)" }}
            whileTap={{ scale: 0.95 }}
            className="w-full sm:w-auto bg-[#052E16] text-white px-10 md:px-16 py-6 md:py-7 rounded-[1.5rem] md:rounded-[2rem] text-[11px] md:text-[12px] font-black uppercase tracking-[0.3em] hover:bg-black transition-all shadow-2xl shadow-[#052E16]/20 flex items-center justify-center gap-4"
          >
            Launch Bridge <Play size={18} fill="currentColor" />
          </MotionLink>
          <MotionLink
            to="/about"
            whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(16, 185, 129, 0.3)" }}
            whileTap={{ scale: 0.95 }}
            className="w-full sm:w-auto bg-[var(--bg-primary)] text-[var(--text-primary)] border-2 border-[var(--border-color)] px-10 md:px-16 py-6 md:py-7 rounded-[1.5rem] md:rounded-[2rem] text-[11px] md:text-[12px] font-black uppercase tracking-[0.3em] hover:bg-[var(--bg-tertiary)] transition-all flex items-center justify-center gap-4"
          >
            Capabilities <ArrowRight size={18} />
          </MotionLink>
        </motion.div>
      </section>

      {/* Verdant Feature Grid */}
      <section className="py-32 md:py-64 bg-[var(--bg-primary)] relative">
        <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#BBF7D0] to-transparent" />
        <div className="max-w-[1400px] mx-auto px-6 md:px-10 text-center mb-20 md:mb-40">
             <h2 className="text-[0.65rem] font-black text-[#16A34A] uppercase tracking-[0.6em] mb-6">Core Philosophy</h2>
             <h3 className="text-4xl md:text-5xl font-black text-[#052E16] tracking-tight leading-tight">Built for Accessibility.<br/>Inspired by Nature.</h3>
        </div>
        <div className="max-w-[1400px] mx-auto px-6 md:px-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
            {features.map((feature, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.02 }}
                whileHover={{ y: -15 }}
                className="relative bg-[var(--bg-secondary)] border border-[var(--border-color)] p-10 md:p-16 rounded-[3rem] md:rounded-[4rem] shadow-[0_20px_50px_-15px_rgba(22,163,74,0.06)] hover:shadow-[0_40px_80px_-20px_rgba(22,163,74,0.12)] transition-all group overflow-hidden"
              >
                <div className="absolute -top-16 -right-16 w-32 md:w-48 h-32 md:h-48 bg-[#F0FDF4] rounded-full blur-[60px] md:blur-[80px] opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className={`w-16 h-16 md:w-20 md:h-20 ${feature.color} ${feature.textColor || 'text-white'} flex items-center justify-center rounded-[1.5rem] md:rounded-[1.75rem] mb-8 md:mb-12 shadow-2xl ${feature.shadow} group-hover:scale-110 transition-transform`}>
                  <feature.icon size={28} strokeWidth={2.5} />
                </div>
                <h4 className="text-2xl md:text-3xl font-black text-[#052E16] mb-6 md:mb-8 tracking-tighter italic">{feature.title}</h4>
                <p className="text-[#052E16]/50 leading-relaxed font-bold text-lg md:text-xl border-l-4 border-[#F0FDF4] pl-6 md:pl-10 group-hover:border-[#16A34A] transition-colors">
                  {feature.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* The Stack Visualization */}
      <section className="py-32 md:py-40 bg-[var(--bg-primary)] relative">
        <div className="max-w-[1400px] mx-auto px-6 md:px-10 text-center mb-16 md:mb-24">
             <h2 className="text-[0.65rem] font-black text-[#16A34A] uppercase tracking-[0.6em] mb-6">System Visualization</h2>
             <h3 className="text-4xl md:text-5xl font-black text-[var(--text-primary)] tracking-tight">The Verdant Stack</h3>
        </div>
        <div className="hidden md:block">
          <NeuralTechField />
        </div>
        {NeuralTechField && (
           <div className="md:hidden px-6 flex flex-col gap-6">
              <div className="bg-[#F0FDF4] p-10 rounded-[3rem] border border-[#BBF7D0]/30 text-center">
                 <Sparkles className="mx-auto text-[#16A34A] mb-4" size={32} />
                 <p className="text-[10px] font-black uppercase tracking-[0.3em] text-[#052E16]/40">Active Neural Core</p>
                 <h4 className="text-xl font-black text-[#052E16] mt-2 italic">Optimized for Mobile Vision</h4>
              </div>
           </div>
        )}
      </section>

      <footer className="py-24 bg-[var(--bg-primary)] border-t border-[var(--border-color)]">
        <div className="max-w-[1400px] mx-auto px-10 flex flex-col items-center gap-8">
           <Link to="/" onClick={() => window.scrollTo(0, 0)}>
             <img src="/logo.png" className="h-12 object-contain grayscale opacity-20 hover:grayscale-0 hover:opacity-100 transition-all duration-700" alt="SignBridge" />
           </Link>
           <span className="text-[10px] font-black text-gray-200 uppercase tracking-[0.5em]">Neural Interpreter 4.0</span>
        </div>
      </footer>
    </motion.div>
  );
}

export default Home;
