import React from 'react';
import { motion } from 'framer-motion';
import {
  Check, Camera, Scan, Brain, Workflow, Volume2,
  ArrowRight, Github, Linkedin, Cpu, Shield, Globe,
  Terminal, Sparkles
} from 'lucide-react';

function About() {
  const steps = [
    { title: 'Capture', desc: 'Webcam or depth camera', icon: Camera, color: 'text-emerald-500' },
    { title: 'Detect', desc: 'MediaPipe 21 keypoints', icon: Scan, color: 'text-blue-500' },
    { title: 'Classify', desc: 'TF CNN + PyTorch LSTM', icon: Brain, color: 'text-purple-500' },
    { title: 'Merge', desc: 'Ensemble controller', icon: Workflow, color: 'text-amber-500' },
    { title: 'Output', desc: 'Text + Voice overlay', icon: Volume2, color: 'text-emerald-500' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen bg-[var(--bg-primary)] font-inter pb-32 transition-colors duration-300"
    >
      {/* Hero Section */}
      <section className="bg-[var(--bg-secondary)] border-b border-[var(--border-color)] pt-40 pb-32 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-bl from-emerald-100/40 to-transparent rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-gradient-to-tr from-emerald-50/40 to-transparent rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-[1200px] mx-auto px-6 text-center relative z-10">
          <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-50/50 text-emerald-700 rounded-full text-[10px] font-black uppercase tracking-[0.3em] mb-10 border border-emerald-100 shadow-sm">
            <Sparkles size={14} /> Mission Protocol
          </motion.div>
          <motion.h1 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.02 }} className="text-7xl md:text-8xl font-black tracking-tighter text-[var(--text-primary)] mb-10 leading-[0.95]">
            Where Signs <br /> <span className="text-[#10b981] italic">Become Voice.</span>
          </motion.h1>
          <motion.p initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.04 }} className="text-2xl text-gray-500 font-medium leading-relaxed max-w-4xl mx-auto tracking-tight">
            SignBridge AI is a high-performance, real-time ASL translation engine. Designed to seamlessly bridge the communication gap by converting organic sign language into spoken and textual context instantly.
          </motion.p>
        </div>
      </section>

      {/* Steps Pipeline */}
      <section className="max-w-[1200px] mx-auto px-6 py-40">
        <div className="text-center mb-24">
          <h2 className="text-[0.7rem] font-black text-[#16A34A] uppercase tracking-[0.5em] mb-6">Architecture</h2>
          <h3 className="text-5xl font-black text-[#052E16] tracking-tight">The Neural Pipeline.</h3>
        </div>

        <div className="relative">
          <div className="absolute top-[80px] left-[10%] right-[10%] h-px border-t border-dashed border-[#16A34A]/20 hidden lg:block" />

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-16 relative z-10">
            {steps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.02 }}
                className="text-center group"
              >
                <div className="relative mb-10 inline-block">
                  <div className={`w-40 h-40 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-[3rem] shadow-sm flex items-center justify-center ${step.color} transition-all duration-700 group-hover:scale-110 group-hover:shadow-2xl group-hover:shadow-emerald-500/10 group-hover:border-emerald-100`}>
                    <step.icon size={48} strokeWidth={1} />
                  </div>
                  <div className="absolute -top-4 -right-4 w-12 h-12 bg-white text-[#052E16] rounded-full flex items-center justify-center text-xs font-black shadow-xl border border-gray-50 transition-all group-hover:bg-[#052E16] group-hover:text-white">
                    {i + 1}
                  </div>
                </div>

                <h3 className="text-2xl font-black text-[var(--text-primary)] mb-4 tracking-tighter">{step.title}</h3>
                <p className="text-[0.65rem] font-black text-gray-400 uppercase tracking-[0.3em] leading-relaxed px-2 opacity-60">
                  {step.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Core Philosophies Grid */}
      <section className="max-w-[1200px] mx-auto px-6 mb-40">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          <div className="bg-[var(--bg-secondary)] p-12 rounded-[3.5rem] shadow-sm border border-[var(--border-color)] hover:shadow-2xl hover:shadow-emerald-500/5 transition-all duration-500">
            <div className="w-16 h-16 bg-emerald-50 text-emerald-600 flex items-center justify-center rounded-2xl mb-10 shadow-inner">
              <Cpu size={28} />
            </div>
            <h3 className="text-3xl font-black text-[#052E16] mb-6 tracking-tight">Micro-Latency Engine</h3>
            <p className="text-gray-500 leading-relaxed font-medium text-lg tracking-tight">
              Our Vision Engine tracks 21 unique hand keypoints in absolute real-time. By utilizing MediaPipe alongside an optimized LSTM Core, we map skeletal coordinates to neural tensors in milliseconds.
            </p>
          </div>

          <div className="bg-[var(--bg-secondary)] p-12 rounded-[3.5rem] shadow-sm border border-[var(--border-color)] hover:shadow-2xl hover:shadow-emerald-500/5 transition-all duration-500">
            <div className="w-16 h-16 bg-blue-50/10 text-blue-600 flex items-center justify-center rounded-2xl mb-10 shadow-inner">
              <Shield size={28} />
            </div>
            <h3 className="text-3xl font-black text-[var(--text-primary)] mb-6 tracking-tight">Zero-Trust Privacy</h3>
            <p className="text-gray-500 leading-relaxed font-medium text-lg tracking-tight">
              We strictly adhere to data sovereignty. All optical interpretation happens locally on your machine's hardware. Absolutely no video feeds or images are ever transmitted to external servers.
            </p>
          </div>

          <div className="bg-[#052E16] p-12 rounded-[3.5rem] shadow-2xl shadow-[#052E16]/40 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-64 h-64 bg-[#16A34A] rounded-full blur-[80px] opacity-20 group-hover:opacity-40 transition-opacity duration-700" />
            <div className="w-16 h-16 bg-white/10 text-[#BBF7D0] flex items-center justify-center rounded-2xl mb-10 border border-white/5 relative z-10 shadow-2xl">
              <Globe size={28} />
            </div>
            <h3 className="text-3xl font-black text-white mb-6 tracking-tight relative z-10">Global Accessibility</h3>
            <p className="text-emerald-100/60 leading-relaxed font-medium text-lg tracking-tight relative z-10">
              Technology should dismantle barriers, not build them. SignBridge is built to be an open, highly accessible tool requiring only a standard webcam to unlock professional-grade translation.
            </p>
          </div>
        </div>
      </section>

      {/* Creator Banner */}
      <section className="max-w-[1300px] mx-auto px-6 mb-32">
        <div className="bg-[var(--bg-secondary)] rounded-[4.5rem] p-16 md:p-24 shadow-2xl shadow-emerald-900/5 border border-[var(--border-color)] relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-16">
          <div className="absolute -left-20 -bottom-20 w-[400px] h-[400px] bg-emerald-50 rounded-full blur-[100px] pointer-events-none opacity-50" />

          <div className="relative z-10 max-w-2xl text-center md:text-left">
            <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gray-50 text-gray-500 rounded-full text-[10px] font-black uppercase tracking-[0.4em] mb-8 border border-gray-100">
              <Terminal size={14} /> System Architect
            </div>
            <h2 className="text-5xl md:text-7xl font-black text-[#052E16] mb-8 tracking-tighter leading-tight">Built by <br /> <span className="text-[#10b981]">Zeeshan Hussain.</span></h2>
            <p className="text-gray-500 font-medium text-xl leading-relaxed mb-12 tracking-tight">
              The creator and lead developer behind the SignBridge inference engine. Dedicated to pushing the boundaries of Neural AI, Computer Vision, and highly-performant React architectures.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 justify-center md:justify-start">
              {/* GitHub Link */}
              <motion.a
                href="https://github.com/zeeshan-create"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05, boxShadow: "0 0 25px rgba(16, 185, 129, 0.4)" }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-3 bg-[#052E16] text-white px-8 py-4 rounded-[1.5rem] text-[0.75rem] font-black uppercase tracking-widest hover:bg-[#16A34A] transition-all w-full sm:w-auto justify-center shadow-lg shadow-[#052E16]/20"
              >
                <Github size={18} /> GitHub Profile
              </motion.a>

              {/* LinkedIn Link */}
              <motion.a
                href="https://www.linkedin.com/in/zeeshan-hussain-32b9a2286/"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05, boxShadow: "0 0 25px rgba(10, 102, 194, 0.4)" }}
                whileTap={{ scale: 0.95 }}
                className="flex items-center gap-3 bg-[#0A66C2] text-white px-8 py-4 rounded-[1.5rem] text-[0.75rem] font-black uppercase tracking-widest hover:bg-[#004182] transition-all w-full sm:w-auto justify-center shadow-lg shadow-[#0A66C2]/20"
              >
                <Linkedin size={18} /> LinkedIn
              </motion.a>
            </div>
          </div>

          {/* Visual Element for Creator */}
          <div className="relative z-10 hidden lg:block">
            <div className="w-64 h-64 bg-gray-50 rounded-full border-[8px] border-white shadow-2xl flex items-center justify-center p-8 relative">
              <div className="absolute inset-0 rounded-full border border-gray-100 animate-[spin_10s_linear_infinite]" />
              <img src="/logo.png" className="w-full h-full object-contain filter grayscale opacity-40 hover:grayscale-0 hover:opacity-100 transition-all duration-500 scale-75" alt="SignBridge" />
            </div>
          </div>
        </div>
      </section>

    </motion.div>
  );
}

export default About;
