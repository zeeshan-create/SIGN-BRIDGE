import React from 'react';
import { motion } from 'framer-motion';
import LexiconModule from './LexiconModule';
import { Type, Cpu, Globe } from 'lucide-react';

export default function Converter({ socket }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="min-h-screen pt-20 bg-[var(--bg-primary)] transition-colors duration-300"
    >
      {/* Premium Header Section */}
      <div className="bg-[var(--nav-bg)] backdrop-blur-xl border-b border-[var(--border-color)] py-16 transition-colors duration-300">
        <div className="max-w-[1400px] mx-auto px-10">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[var(--bg-tertiary)] text-[#10b981] border border-[var(--border-color)] rounded-xl flex items-center justify-center shadow-sm">
                  <Type size={20} />
                </div>
                <h1 className="text-3xl font-black text-[var(--text-primary)] tracking-tight">ASL Lexicon Converter</h1>
              </div>
              <p className="text-[var(--text-secondary)] font-medium max-w-xl">
                A focused workspace for accurate English-to-ASL translation using an AI-powered sequential pipeline.
              </p>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3 bg-[var(--bg-tertiary)] px-4 py-2 rounded-xl text-[#10b981] text-[10px] font-black uppercase tracking-widest border border-[var(--border-color)]">
                <Cpu size={14} /> Neural Link: Active
              </div>
              <div className="flex items-center gap-3 bg-[var(--bg-tertiary)] px-4 py-2 rounded-xl text-[#3b82f6] text-[10px] font-black uppercase tracking-widest border border-[var(--border-color)]">
                <Globe size={14} /> Translation Engine: v1.0
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Workstation */}
      <main className="relative">
        <div className="absolute top-0 left-0 w-full h-screen bg-gradient-to-b from-[var(--bg-primary)] to-transparent pointer-events-none" />
        <div className="relative z-10">
           <LexiconModule socket={socket} />
        </div>
      </main>

      {/* Background Tech HUD Details */}
      <div className="fixed bottom-10 left-10 opacity-20 pointer-events-none hidden xl:block z-0">
        <div className="space-y-1">
          <p className="text-[10px] font-black text-[var(--text-primary)] uppercase tracking-widest opacity-60">Structure Mapping Engine v1</p>
          <p className="text-[8px] font-bold text-[var(--text-tertiary)] uppercase tracking-widest">ASL Grammar Restructuring module</p>
        </div>
      </div>
    </motion.div>
  );
}
